"""CLI: full per-session extraction (trials + spikes + covariates).

    python -m ibl_pipeline.extract --regions VISp,CP --max-sessions-per-region 3

One directory per session under `data/sessions/<eid>/`, in the same schema for
every region. Re-running skips any session whose manifest is already complete, so
an interrupted run resumes rather than re-downloading.

Scope note: this is the expensive half of the pipeline (spike trains are ~10^2 MB
per insertion). The coverage/QC numbers in the pipeline document come from
`run_survey`, which needs none of this.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .behavior import TrialExtractionError, extract_session_behavior
from .covariates import extract_covariates
from .insertions import build_insertion_table
from .one_client import get_one
from .spikes import (
    SpikeExtractionError,
    load_spike_trains,
    prestim_spike_counts,
    tag_target_region,
)
from .survey import cached_cluster_table

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 2


def session_dir(eid: str) -> Path:
    return config.SESSIONS_DIR / str(eid)


def is_complete(eid: str) -> bool:
    """True when a previous run finished this session (idempotency guard)."""
    manifest = session_dir(eid) / MANIFEST_NAME
    if not manifest.exists():
        return False
    try:
        record = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "complete" and record.get("schema_version") == SCHEMA_VERSION


def _prune_spike_cache(one: Any, eid: str) -> int:
    """Delete cached spike trains for a session after counts are computed.

    The counts are the deliverable; the raw trains are ~100x larger and would
    exhaust the disk after a handful of sessions. Cluster/trial/covariate files
    are small and stay cached.
    """
    try:
        path = Path(one.eid2path(eid))
    except Exception:  # noqa: BLE001
        return 0
    freed = 0
    # rglob, not a fixed-depth glob: revisioned sessions nest one level deeper
    # (alf/probe00/pykilosort/#2024-05-06#/spikes.times.npy), and a fixed pattern
    # silently matches nothing there -- pruning would quietly stop working on
    # exactly the sessions whose files are largest.
    for spike_file in path.rglob("spikes.*.npy"):
        try:
            freed += spike_file.stat().st_size
            spike_file.unlink()
        except OSError:
            continue
    return freed


def extract_session(
    one: Any,
    eid: str,
    session_insertions: pd.DataFrame,
    prune_cache: bool = False,
) -> dict[str, Any]:
    """Extract one session into `data/sessions/<eid>/`; returns its manifest."""
    started = time.time()
    out_dir = session_dir(eid)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = session_insertions.iloc[0]
    manifest: dict[str, Any] = {
        "eid": str(eid),
        "schema_version": SCHEMA_VERSION,
        "subject": info["subject"],
        "lab": info["lab"],
        "task_protocol": info["task_protocol"],
        "stage": info["stage"],
        "prestim_window_s": list(config.PRESTIM_WINDOW_S),
        "qc_label_pass": config.QC_LABEL_PASS,
        "target_regions": sorted(set(session_insertions["target_region"])),
        "status": "failed",
        "issues": [],
    }

    # --- trials (guard first: no trials means nothing else is meaningful) ---
    try:
        trials = extract_session_behavior(one, eid, info["task_protocol"])
    except TrialExtractionError as exc:
        manifest["issues"].append(str(exc))
        (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))
        return manifest

    # Spike counts and covariates are only defined where stimOn is logged.
    trials = trials[trials["stimOn_times"].notna()].reset_index(drop=True)
    if trials.empty:
        manifest["issues"].append("no trials with a valid stimOn_times")
        (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))
        return manifest

    trials.to_parquet(out_dir / "trials.parquet", index=False)
    manifest["n_trials"] = int(len(trials))
    manifest["n_trials_biased_block"] = int(trials["probabilityLeft"].isin([0.2, 0.8]).sum())
    events = trials["stimOn_times"].to_numpy(dtype=float)

    # --- spikes, per insertion ---
    cluster_frames: list[pd.DataFrame] = []
    count_blocks: list[np.ndarray] = []
    per_insertion: list[dict[str, Any]] = []

    for pid, probe_rows in session_insertions.groupby("pid"):
        regions = sorted(set(probe_rows["target_region"]))
        record: dict[str, Any] = {"pid": str(pid), "probe": probe_rows.iloc[0]["probe"],
                                  "target_regions": regions}
        try:
            table = cached_cluster_table(one, str(pid))
        except SpikeExtractionError as exc:
            record["status"] = f"skipped: {exc}"
            per_insertion.append(record)
            manifest["issues"].append(f"{pid}: {exc}")
            continue

        # One row per cluster, tagged with whichever target region it sits in.
        tagged = pd.concat([tag_target_region(table, r) for r in regions], ignore_index=True)
        keep = tagged[tagged["in_target_region"] & tagged["qc_pass"]].drop_duplicates(
            subset=["pid", "cluster_id"])
        record["n_clusters_total"] = int(len(table))
        record["n_clusters_qc_pass_in_target"] = int(len(keep))

        if keep.empty:
            # Reported, not silently dropped: an insertion listed as reaching the
            # region but yielding no good units there is a real coverage loss.
            record["status"] = "no QC-passing clusters in target region"
            per_insertion.append(record)
            manifest["issues"].append(
                f"{pid}: 0 QC-passing clusters in {regions} (of {len(table)} sorted)")
            continue

        try:
            spike_times, spike_clusters = load_spike_trains(one, str(pid))
        except SpikeExtractionError as exc:
            record["status"] = f"spike trains unavailable: {exc}"
            per_insertion.append(record)
            manifest["issues"].append(f"{pid}: {exc}")
            continue

        cluster_ids = keep["cluster_id"].to_numpy(dtype=int)
        counts = prestim_spike_counts(
            spike_times, spike_clusters, cluster_ids, events, config.PRESTIM_WINDOW_S)
        count_blocks.append(counts)
        cluster_frames.append(keep)
        record["status"] = "ok"
        record["n_spikes"] = int(spike_times.size)
        per_insertion.append(record)

        if prune_cache:
            record["cache_pruned_bytes"] = _prune_spike_cache(one, eid)

    manifest["insertions"] = per_insertion

    if cluster_frames:
        clusters = pd.concat(cluster_frames, ignore_index=True)
        clusters.to_parquet(out_dir / "clusters.parquet", index=False)
        counts = np.concatenate(count_blocks, axis=1)
        np.save(out_dir / "prestim_counts.npy", counts)
        manifest["n_clusters"] = int(len(clusters))
        manifest["counts_shape"] = list(counts.shape)
    else:
        manifest["n_clusters"] = 0
        manifest["counts_shape"] = [int(len(trials)), 0]
        manifest["issues"].append("no usable clusters in any target region for this session")

    # --- covariates (never fatal) ---
    covariates, covariate_status = extract_covariates(one, eid, events)
    covariates.to_parquet(out_dir / "covariates.parquet", index=False)
    manifest["covariate_status"] = covariate_status
    for name, state in covariate_status.items():
        if state != "ok":
            manifest["issues"].append(f"covariate {name}: {state}")

    manifest["status"] = "complete" if manifest["n_clusters"] > 0 else "complete_no_units"
    manifest["elapsed_s"] = round(time.time() - started, 1)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))
    return manifest


def select_sessions(
    insertions: pd.DataFrame, regions: tuple[str, ...], max_per_region: int | None
) -> pd.DataFrame:
    """Choose which sessions to extract, capped per region.

    Sessions covering the most target regions are taken first: they carry the
    cross-region comparisons, which a per-region cap would otherwise fragment.
    """
    included = insertions[insertions["included"] & insertions["target_region"].isin(regions)]
    if included.empty:
        return included

    # Always return breadth-ranked, even when uncapped: `--max-sessions` truncates
    # this order, so ranking here is what makes "give me the N richest sessions"
    # actually pick the richest rather than whichever happened to sort first.
    breadth = included.groupby("eid")["target_region"].nunique().rename("n_regions")
    ranked = included.merge(breadth, on="eid").sort_values(
        ["n_regions", "eid"], ascending=[False, True])

    if max_per_region is None:
        return ranked.drop(columns="n_regions")

    chosen: set[str] = set()
    per_region: dict[str, int] = {r: 0 for r in regions}
    for eid, rows in ranked.groupby("eid", sort=False):
        eid_regions = set(rows["target_region"])
        if any(per_region[r] < max_per_region for r in eid_regions):
            chosen.add(str(eid))
            for r in eid_regions:
                per_region[r] += 1
    return ranked[ranked["eid"].isin(chosen)].drop(columns="n_regions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", default=",".join(config.TARGET_REGIONS))
    parser.add_argument("--max-sessions-per-region", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="hard cap on total sessions this run")
    parser.add_argument("--eids", default=None,
                        help="comma-separated session ids to extract, overriding "
                             "the automatic breadth-ranked selection")
    parser.add_argument("--prune-cache", action="store_true",
                        help="delete raw spike trains after counting (bounds disk use)")
    parser.add_argument("--force", action="store_true", help="re-extract completed sessions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    regions = tuple(r.strip() for r in args.regions.split(",") if r.strip())
    unknown = [r for r in regions if r not in config.REGION_LEVELS]
    if unknown:
        parser.error(f"unknown target region(s): {unknown}")

    one = get_one()
    survey_file = config.SURVEY_DIR / "insertions.parquet"
    insertions = (pd.read_parquet(survey_file) if survey_file.exists()
                  else build_insertion_table(one, regions))

    if args.eids:
        # An explicit list pins the pilot set, so a later re-run extracts the same
        # sessions even if the ranking shifts as more of the survey lands.
        wanted = [e.strip() for e in args.eids.split(",") if e.strip()]
        selected = insertions[insertions["included"] & insertions["eid"].isin(wanted)]
        unknown_eids = set(wanted) - set(selected["eid"])
        if unknown_eids:
            parser.error(f"eid(s) not in the filtered insertion table: {sorted(unknown_eids)}")
        eids = wanted
    else:
        selected = select_sessions(insertions, regions, args.max_sessions_per_region)
        eids = list(dict.fromkeys(selected["eid"]))
    if args.max_sessions is not None:
        eids = eids[: args.max_sessions]

    todo = [e for e in eids if args.force or not is_complete(e)]
    print(f"{len(eids)} sessions selected; {len(eids) - len(todo)} already complete; "
          f"{len(todo)} to extract", flush=True)
    if args.dry_run:
        for eid in todo:
            print("  would extract", eid)
        return 0

    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for i, eid in enumerate(todo, start=1):
        rows = selected[selected["eid"] == eid]
        print(f"[{i}/{len(todo)}] {eid} ({rows.iloc[0]['subject']}, "
              f"{sorted(set(rows['target_region']))})", flush=True)
        try:
            manifest = extract_session(one, eid, rows, prune_cache=args.prune_cache)
        except Exception as exc:  # noqa: BLE001 - one bad session must not kill the run
            failed += 1
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        if manifest["status"] == "failed":
            failed += 1
        print(f"    status={manifest['status']} trials={manifest.get('n_trials')} "
              f"clusters={manifest.get('n_clusters')} issues={len(manifest['issues'])}",
              flush=True)

    print(f"\ndone: {len(todo) - failed}/{len(todo)} sessions extracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
