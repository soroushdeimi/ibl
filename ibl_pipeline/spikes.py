"""Spike-sorting extraction: per-cluster region labels, QC, pre-stimulus counts.

Split deliberately into a cheap metadata path (`load_cluster_table`, a few
hundred kB per insertion) and an expensive path (`load_spike_trains`, hundreds of
MB), because the QC survey needs only the former across the whole brain-wide map.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from . import config


class SpikeExtractionError(RuntimeError):
    """Raised when an insertion's spike sorting is missing or unusable."""


@lru_cache(maxsize=1)
def _brain_regions() -> Any:
    from iblatlas.regions import BrainRegions
    return BrainRegions()


@lru_cache(maxsize=64)
def region_atlas_ids(region: str) -> frozenset[int]:
    """All Allen CCF ids belonging to `region`, including its subdivisions.

    Cortical areas are annotated per layer (VISp -> VISp1, VISp2/3, ...) and
    thalamic ones per subnucleus (LGd -> LGd-co, ...), so matching the acronym
    literally would discard almost every cortical cluster. Matching on the
    descendant id set is the only correct test.
    """
    regions = _brain_regions()
    ids = regions.acronym2id(region)
    if len(ids) == 0:
        raise SpikeExtractionError(f"unknown Allen acronym: {region!r}")
    return frozenset(int(i) for i in regions.descendants(ids=ids)["id"])


def _spike_sorting_loader(one: Any, pid: str) -> Any:
    from brainbox.io.one import SpikeSortingLoader
    return SpikeSortingLoader(pid=pid, one=one)


def load_cluster_table(one: Any, pid: str) -> pd.DataFrame:
    """Per-cluster table for one insertion: id, channel, CCF acronym, QC label.

    Cheap: pulls clusters.* and channels.* only, never the spike trains.
    """
    loader = _spike_sorting_loader(one, pid)
    try:
        clusters = loader.load_spike_sorting_object("clusters")
        channels = loader.load_channels()
    except Exception as exc:  # noqa: BLE001
        raise SpikeExtractionError(f"{pid}: could not load cluster/channel data ({exc})") from exc

    if clusters is None or "channels" not in clusters:
        raise SpikeExtractionError(f"{pid}: clusters object has no channel mapping")

    cluster_channels = np.asarray(clusters["channels"]).astype(int)
    n_clusters = cluster_channels.size
    if n_clusters == 0:
        raise SpikeExtractionError(f"{pid}: spike sorting contains zero clusters")

    acronyms = np.asarray(channels.get("acronym", np.full(0, "")), dtype=object)
    atlas_ids = np.asarray(channels.get("atlas_id", np.full(0, np.nan)))
    if acronyms.size == 0 or atlas_ids.size == 0:
        raise SpikeExtractionError(f"{pid}: no histology (channel region labels) available")

    # A cluster inherits the region of its peak channel; clip guards against the
    # rare sorting where a cluster references a channel outside the map.
    idx = np.clip(cluster_channels, 0, acronyms.size - 1)
    table = pd.DataFrame({
        "pid": str(pid),
        "cluster_id": np.arange(n_clusters, dtype=int),
        "channel": cluster_channels,
        "acronym": acronyms[idx],
        "atlas_id": atlas_ids[idx].astype("int64", copy=False),
    })

    metrics = clusters.get("metrics")
    if metrics is None or "label" not in getattr(metrics, "columns", []):
        # Without labels we cannot honour "QC-passing only", so refuse rather
        # than quietly returning MUA/noise as if it had passed.
        raise SpikeExtractionError(f"{pid}: clusters.metrics has no QC `label` column")

    label = pd.Series(np.asarray(metrics["label"], dtype=float))
    if len(label) != n_clusters:
        raise SpikeExtractionError(
            f"{pid}: QC label length {len(label)} != cluster count {n_clusters}")
    table["qc_label"] = label.to_numpy()
    table["qc_pass"] = table["qc_label"] >= config.QC_LABEL_PASS

    if "depths" in clusters:
        depths = np.asarray(clusters["depths"], dtype=float)
        if depths.size == n_clusters:
            table["depth_um"] = depths
    return table


def tag_target_region(table: pd.DataFrame, region: str) -> pd.DataFrame:
    """Flag clusters that fall inside `region` (hierarchy-aware)."""
    ids = region_atlas_ids(region)
    out = table.copy()
    out["target_region"] = region
    out["in_target_region"] = out["atlas_id"].isin(ids)
    return out


def load_spike_trains(one: Any, pid: str) -> tuple[np.ndarray, np.ndarray]:
    """Spike times and cluster assignments for one insertion (large download).

    Deliberately bypasses `SpikeSortingLoader.load_spike_sorting_object`: that
    method takes an `attribute=` argument but never uses it, deriving attributes
    from `_get_attributes`, which always unions the default set (times, clusters,
    amps, depths). Going through it therefore downloads `spikes.amps` and
    `spikes.depths` -- roughly 2.7x the bytes -- for arrays we immediately
    discard. `one.load_object` does honour `attribute`, so we resolve the
    collection through the loader and then fetch only the two arrays we need.
    """
    loader = _spike_sorting_loader(one, pid)
    try:
        collection = loader.collection or loader._get_spike_sorting_collection()
        spikes = one.load_object(
            loader.eid, obj="spikes", collection=collection,
            attribute=["times", "clusters"], revision=loader.revision,
        )
    except Exception as exc:  # noqa: BLE001
        # Fall back to the loader's own path rather than lose the session over an
        # upstream API change in a private helper.
        try:
            spikes = loader.load_spike_sorting_object("spikes")
        except Exception:  # noqa: BLE001
            raise SpikeExtractionError(
                f"{pid}: could not load spike trains ({exc})") from exc

    if spikes is None or "times" not in spikes or "clusters" not in spikes:
        raise SpikeExtractionError(f"{pid}: spike trains missing times/clusters")
    times = np.asarray(spikes["times"], dtype=float)
    assignments = np.asarray(spikes["clusters"], dtype=int)
    if times.size != assignments.size:
        raise SpikeExtractionError(f"{pid}: spikes.times and spikes.clusters length mismatch")
    return times, assignments


def prestim_spike_counts(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    cluster_ids: np.ndarray,
    event_times: np.ndarray,
    window: tuple[float, float] = config.PRESTIM_WINDOW_S,
) -> np.ndarray:
    """Spike counts per (trial, cluster) in `window` relative to each event.

    Pure numpy so it can be unit-tested without touching the network. Uses
    searchsorted on the (already sorted) spike times per cluster, which is
    O(n_trials log n_spikes) rather than the O(n_trials * n_spikes) a mask loop
    would cost -- this matters at ~30M spikes per insertion.
    """
    start_offset, end_offset = window
    if end_offset <= start_offset:
        raise ValueError(f"window end must exceed start, got {window}")

    events = np.asarray(event_times, dtype=float)
    counts = np.full((events.size, cluster_ids.size), np.nan, dtype=np.float32)
    valid = np.isfinite(events)
    if not valid.any():
        return counts

    order = np.argsort(spike_times, kind="stable")
    times_sorted, clusters_sorted = spike_times[order], spike_clusters[order]

    for col, cluster_id in enumerate(cluster_ids):
        cluster_times = times_sorted[clusters_sorted == cluster_id]
        if cluster_times.size == 0:
            counts[valid, col] = 0
            continue
        starts = np.searchsorted(cluster_times, events[valid] + start_offset, side="left")
        ends = np.searchsorted(cluster_times, events[valid] + end_offset, side="right")
        counts[valid, col] = (ends - starts).astype(np.float32)
    return counts
