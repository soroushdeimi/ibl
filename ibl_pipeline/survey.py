"""Descriptive / QC survey across the target regions (analyses 1-5).

Metadata + cluster-level only: no spike trains are downloaded here, which is what
makes it affordable to run over every matching session in the public release.
Each insertion's cluster table is cached to disk, so a re-run resumes instead of
re-fetching.
"""
from __future__ import annotations

import concurrent.futures as futures
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import config
from .behavior import TrialExtractionError, load_trials
from .spikes import SpikeExtractionError, load_cluster_table, region_atlas_ids

CLUSTER_CACHE = config.SURVEY_DIR / "clusters"
TRIAL_CACHE = config.SURVEY_DIR / "trial_counts"


def _ensure_dirs() -> None:
    for path in (config.SURVEY_DIR, CLUSTER_CACHE, TRIAL_CACHE):
        path.mkdir(parents=True, exist_ok=True)


def cached_cluster_table(one: Any, pid: str) -> pd.DataFrame:
    """Cluster table for one insertion, cached on disk (idempotent re-runs)."""
    _ensure_dirs()
    cache_file = CLUSTER_CACHE / f"{pid}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    table = load_cluster_table(one, pid)
    table.to_parquet(cache_file, index=False)
    return table


def collect_cluster_tables(
    one: Any, pids: Iterable[str], max_workers: int = 4
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch cluster tables for many insertions; returns (table, failures).

    Failures are collected rather than raised: one insertion without histology
    must not abort a survey of several hundred.
    """
    _ensure_dirs()
    pid_list = list(dict.fromkeys(pids))
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}

    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        jobs = {pool.submit(cached_cluster_table, one, pid): pid for pid in pid_list}
        for done in futures.as_completed(jobs):
            pid = jobs[done]
            try:
                frames.append(done.result())
            except (SpikeExtractionError, Exception) as exc:  # noqa: BLE001
                failures[pid] = f"{type(exc).__name__}: {exc}"

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return table, failures


def cached_trial_counts(one: Any, eid: str) -> dict[str, Any]:
    """Trial counts for one session, cached on disk."""
    _ensure_dirs()
    cache_file = TRIAL_CACHE / f"{eid}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    frame = load_trials(one, eid)
    n_total = int(len(frame))
    n_valid = int((frame["choice"] != 0).sum())
    block_values = frame["probabilityLeft"].dropna().unique()
    record = {
        "eid": str(eid),
        "n_trials": n_total,
        "n_trials_valid": n_valid,
        "n_trials_omission": n_total - n_valid,
        # Only the 0.2/0.8 blocks carry a real prior; the opening 0.5 block is
        # neutral and cannot contribute to a prior-decoding analysis.
        "n_trials_biased_block": int(frame["probabilityLeft"].isin([0.2, 0.8]).sum()),
        "block_values": sorted(float(v) for v in block_values)[:8],
        "n_trials_with_stim_on": int(frame["stimOn_times"].notna().sum()),
    }
    cache_file.write_text(json.dumps(record))
    return record


def collect_trial_counts(
    one: Any, eids: Iterable[str], max_workers: int = 4
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Trial counts for many sessions; returns (table, failures)."""
    eid_list = list(dict.fromkeys(eids))
    records: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        jobs = {pool.submit(cached_trial_counts, one, eid): eid for eid in eid_list}
        for done in futures.as_completed(jobs):
            eid = jobs[done]
            try:
                records.append(done.result())
            except (TrialExtractionError, Exception) as exc:  # noqa: BLE001
                failures[eid] = f"{type(exc).__name__}: {exc}"
    return pd.DataFrame(records), failures


def annotate_clusters_by_region(
    clusters: pd.DataFrame, insertions: pd.DataFrame
) -> pd.DataFrame:
    """Join clusters onto (insertion, target region) pairs, hierarchy-aware.

    A cluster counts toward a region only if its own CCF id is that region or one
    of its subdivisions -- an insertion reaching VISp also passes through other
    areas, and those units are not VISp units.
    """
    if clusters.empty:
        return pd.DataFrame()

    merged = insertions[["target_region", "pid", "eid", "subject", "lab"]].merge(
        clusters, on="pid", how="inner")
    if merged.empty:
        return merged

    id_sets = {r: region_atlas_ids(r) for r in merged["target_region"].unique()}
    merged["in_target_region"] = [
        int(atlas_id) in id_sets[region]
        for region, atlas_id in zip(merged["target_region"], merged["atlas_id"])
    ]
    return merged


def region_summary(
    insertions: pd.DataFrame,
    region_clusters: pd.DataFrame,
    trial_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Analyses 1, 2, 3 and 4: coverage, cluster yield, trial counts, power flags."""
    rows = []
    for region in config.TARGET_REGIONS:
        included = insertions[(insertions["target_region"] == region) & insertions["included"]]

        in_region = pd.DataFrame()
        if not region_clusters.empty:
            in_region = region_clusters[
                (region_clusters["target_region"] == region)
                & region_clusters["in_target_region"]
            ]
        good = in_region[in_region["qc_pass"]] if not in_region.empty else in_region

        # An insertion counts as covering the region only if it actually yielded a
        # QC-passing unit there: a trajectory clipping the border gives histology
        # hits but no usable units, and counting those would overstate coverage.
        productive_pids = set(good["pid"].unique()) if not good.empty else set()
        productive = included[included["pid"].isin(productive_pids)]

        per_insertion = good.groupby("pid").size() if not good.empty else pd.Series(dtype=float)
        eids = set(productive["eid"])
        session_trials = (
            trial_counts[trial_counts["eid"].isin(eids)]
            if not trial_counts.empty else pd.DataFrame())

        n_sessions = int(productive["eid"].nunique())
        n_subjects = int(productive["subject"].nunique())
        flags = []
        if n_subjects < config.MIN_SUBJECTS_FOR_POWER:
            flags.append(f"only {n_subjects} subject(s)")
        if n_sessions < config.MIN_SESSIONS_FOR_POWER:
            flags.append(f"only {n_sessions} session(s)")

        rows.append({
            "region": region,
            "description": config.REGION_LEVELS[region][0],
            "level": config.REGION_LEVELS[region][1],
            "insertions_found": int((insertions["target_region"] == region).sum()),
            "insertions_pass_filter": int(len(included)),
            "insertions_with_good_units": int(len(productive)),
            "n_sessions": n_sessions,
            "n_subjects": n_subjects,
            "n_clusters_in_region": int(len(in_region)),
            "n_clusters_qc_pass": int(len(good)),
            "clusters_per_insertion_median": (
                float(per_insertion.median()) if len(per_insertion) else 0.0),
            "clusters_per_insertion_min": int(per_insertion.min()) if len(per_insertion) else 0,
            "clusters_per_insertion_max": int(per_insertion.max()) if len(per_insertion) else 0,
            "trials_per_session_median": (
                float(session_trials["n_trials_valid"].median()) if len(session_trials) else np.nan),
            "trials_per_session_min": (
                int(session_trials["n_trials_valid"].min()) if len(session_trials) else 0),
            "trials_per_session_max": (
                int(session_trials["n_trials_valid"].max()) if len(session_trials) else 0),
            "underpowered": bool(flags),
            "power_flag": "; ".join(flags),
        })
    return pd.DataFrame(rows)


def coverage_overlap(insertions: pd.DataFrame, region_clusters: pd.DataFrame) -> pd.DataFrame:
    """Analysis 5: sessions with usable units in more than one target region.

    These sessions permit within-animal, within-session cross-region latency
    comparisons, so they are identified explicitly rather than left implicit.
    """
    empty = pd.DataFrame(columns=["eid", "subject", "n_regions", "regions"])
    if region_clusters.empty:
        return empty

    good = region_clusters[region_clusters["in_target_region"] & region_clusters["qc_pass"]]
    if good.empty:
        return empty

    grouped = good.groupby(["eid", "subject"])["target_region"].agg(lambda s: sorted(set(s)))
    frame = grouped.reset_index().rename(columns={"target_region": "regions"})
    frame["n_regions"] = frame["regions"].apply(len)
    frame["regions"] = frame["regions"].apply(lambda r: ",".join(r))
    return frame.sort_values(["n_regions", "eid"], ascending=[False, True]).reset_index(drop=True)


def region_pair_matrix(overlap: pd.DataFrame) -> pd.DataFrame:
    """Session counts for every pair of simultaneously recorded target regions."""
    matrix = pd.DataFrame(0, index=list(config.TARGET_REGIONS),
                          columns=list(config.TARGET_REGIONS), dtype=int)
    if overlap.empty:
        return matrix
    for regions in overlap["regions"]:
        present = [r for r in str(regions).split(",") if r]
        for i, first in enumerate(present):
            for second in present[i:]:
                matrix.loc[first, second] += 1
                if first != second:
                    matrix.loc[second, first] += 1
    return matrix


COVARIATE_CACHE = config.SURVEY_DIR / "covariate_availability"

# Datasets each Stage 3 covariate depends on. Checked by name rather than by
# loading, so the audit costs one metadata call per session instead of a download.
COVARIATE_DATASETS: dict[str, tuple[str, ...]] = {
    "wheel": ("_ibl_wheel.position", "_ibl_wheel.timestamps"),
    "whisker_me": ("leftCamera.ROIMotionEnergy",),
    "pupil": ("_ibl_leftCamera.features",),
    "camera_times": ("_ibl_leftCamera.times",),
}


def cached_covariate_availability(one: Any, eid: str) -> dict[str, Any]:
    """Which Stage 3 covariates exist for one session (metadata only, cached)."""
    _ensure_dirs()
    COVARIATE_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = COVARIATE_CACHE / f"{eid}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    paths = [str(p) for p in one.list_datasets(eid)]
    record: dict[str, Any] = {"eid": str(eid)}
    for name, needles in COVARIATE_DATASETS.items():
        record[name] = all(any(needle in path for path in paths) for needle in needles)
    cache_file.write_text(json.dumps(record))
    return record


def collect_covariate_availability(
    one: Any, eids: Iterable[str], max_workers: int = 4
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Covariate availability across many sessions; returns (table, failures)."""
    eid_list = list(dict.fromkeys(eids))
    records: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        jobs = {pool.submit(cached_covariate_availability, one, eid): eid for eid in eid_list}
        for done in futures.as_completed(jobs):
            eid = jobs[done]
            try:
                records.append(done.result())
            except Exception as exc:  # noqa: BLE001
                failures[eid] = f"{type(exc).__name__}: {exc}"
    return pd.DataFrame(records), failures
