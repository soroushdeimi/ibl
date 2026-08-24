"""CLI: run the multi-region coverage / QC survey (analyses 1-5).

    python -m ibl_pipeline.run_survey [--regions VISp,CP] [--workers 6]

Writes every intermediate table to `data/survey/` so the pipeline document and
any later analysis read the same numbers, and so a re-run resumes from cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import pandas as pd

from . import config
from .insertions import build_insertion_table
from .one_client import get_one
from .survey import (
    annotate_clusters_by_region,
    collect_cluster_tables,
    collect_covariate_availability,
    collect_trial_counts,
    coverage_overlap,
    region_pair_matrix,
    region_summary,
)

INSERTIONS_FILE = config.SURVEY_DIR / "insertions.parquet"


def _load_or_build_insertions(one: Any, regions: tuple[str, ...], refresh: bool) -> pd.DataFrame:
    """Insertion table, rebuilt only when asked (the Alyx query is slow)."""
    config.SURVEY_DIR.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(INSERTIONS_FILE) if INSERTIONS_FILE.exists() else pd.DataFrame()
    have = set(existing["target_region"]) if not existing.empty else set()
    missing = tuple(r for r in regions if r not in have)

    if refresh or not have:
        table = build_insertion_table(one, regions)
    elif missing:
        # Query only the new regions and merge, so adding a region later does not
        # discard (or re-fetch) the ones already surveyed.
        table = pd.concat([existing, build_insertion_table(one, missing)], ignore_index=True)
        table = table.drop_duplicates(subset=["target_region", "pid"], keep="last")
    else:
        return existing[existing["target_region"].isin(regions)]

    table.to_parquet(INSERTIONS_FILE, index=False)
    return table[table["target_region"].isin(regions)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", default=",".join(config.TARGET_REGIONS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh-insertions", action="store_true")
    parser.add_argument("--skip-trials", action="store_true",
                        help="skip per-session trial counts (analysis 3)")
    parser.add_argument("--skip-covariates", action="store_true",
                        help="skip the Stage 3 covariate availability audit")
    args = parser.parse_args(argv)

    regions = tuple(r.strip() for r in args.regions.split(",") if r.strip())
    unknown = [r for r in regions if r not in config.REGION_LEVELS]
    if unknown:
        parser.error(f"unknown target region(s): {unknown}")

    one = get_one()
    config.SURVEY_DIR.mkdir(parents=True, exist_ok=True)

    insertions = _load_or_build_insertions(one, regions, args.refresh_insertions)
    included = insertions[insertions["included"]]
    print(f"[1/5] insertions: {len(insertions)} found, {len(included)} pass the session filter",
          flush=True)

    excluded = insertions[~insertions["included"]]
    exclusion_counts = excluded["exclusion_reason"].value_counts().to_dict()
    print(f"      excluded: {len(excluded)} -> {exclusion_counts}", flush=True)

    print(f"[2/5] cluster tables for {included['pid'].nunique()} insertions ...", flush=True)
    clusters, cluster_failures = collect_cluster_tables(
        one, included["pid"].unique(), max_workers=args.workers)
    print(f"      loaded {len(clusters)} clusters; {len(cluster_failures)} insertions failed",
          flush=True)

    region_clusters = annotate_clusters_by_region(clusters, included)

    trial_counts = pd.DataFrame()
    trial_failures: dict[str, str] = {}
    if not args.skip_trials:
        eids = included["eid"].unique()
        print(f"[3/5] trial counts for {len(eids)} sessions ...", flush=True)
        trial_counts, trial_failures = collect_trial_counts(one, eids, max_workers=args.workers)
        print(f"      loaded {len(trial_counts)} sessions; {len(trial_failures)} failed",
              flush=True)

    covariates = pd.DataFrame()
    covariate_failures: dict[str, str] = {}
    if not args.skip_covariates:
        eids = included["eid"].unique()
        print(f"[4/5] covariate availability for {len(eids)} sessions ...", flush=True)
        covariates, covariate_failures = collect_covariate_availability(
            one, eids, max_workers=args.workers)
        if not covariates.empty:
            have = {c: f"{covariates[c].mean():.0%}" for c in covariates.columns if c != "eid"}
            print(f"      {have}", flush=True)

    print("[5/5] summarising ...", flush=True)
    summary = region_summary(included, region_clusters, trial_counts)
    overlap = coverage_overlap(included, region_clusters)
    pairs = region_pair_matrix(overlap)

    summary.to_csv(config.SURVEY_DIR / "region_summary.csv", index=False)
    overlap.to_csv(config.SURVEY_DIR / "coverage_overlap.csv", index=False)
    pairs.to_csv(config.SURVEY_DIR / "region_pair_matrix.csv")
    if not trial_counts.empty:
        trial_counts.to_csv(config.SURVEY_DIR / "trial_counts.csv", index=False)
    if not covariates.empty:
        covariates.to_csv(config.SURVEY_DIR / "covariate_availability.csv", index=False)
    if not region_clusters.empty:
        region_clusters.to_parquet(config.SURVEY_DIR / "region_clusters.parquet", index=False)

    failures = {
        "excluded_insertions_by_reason": exclusion_counts,
        "cluster_load_failures": cluster_failures,
        "trial_load_failures": trial_failures,
        "covariate_audit_failures": covariate_failures,
    }
    (config.SURVEY_DIR / "failures.json").write_text(json.dumps(failures, indent=1))

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[[
            "region", "n_sessions", "n_subjects", "n_clusters_qc_pass",
            "clusters_per_insertion_median", "trials_per_session_median", "power_flag",
        ]].to_string(index=False))
        print("\nsessions with >1 target region:",
              int((overlap["n_regions"] > 1).sum()) if not overlap.empty else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
