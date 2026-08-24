"""CLI: Stage 1 behavioural model fitting (proposal Q8, Models 1-2 + GLM-HMM gate).

    python -m ibl_pipeline.run_stage1 [--n-sessions 80] [--min-biased-trials 200]

Pulls trials only (no spike sorting) for a diverse pilot set of sessions,
selected from the survey's already-cached insertion/trial-count tables, so
this needs no re-querying of Alyx for session discovery. Writes every
intermediate table to `data/survey/stage1/` so a re-run resumes from cache and
`report.py`-style downstream tooling reads the same numbers this prints.

Pipeline: select sessions -> pull+cache trials -> Model 1 (history GLM, nested
session-grouped CV) -> GLM-HMM (3-state) -> Model 2 (four belief models,
compared with and without the engaged-state restriction from step 5) ->
figures (`plots.py`, written to `data/survey/stage1/figures/`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .behavior import TrialExtractionError, extract_session_behavior
from .behavior_models import (
    BELIEF_MODELS,
    compare_belief_models,
    fit_glm_hmm,
    fit_history_glm_interpretable,
    fit_history_glm_nested_cv,
    history_glm_design,
    label_engaged_state,
)
from .one_client import get_one

STAGE1_DIR = config.SURVEY_DIR / "stage1"
TRIALS_CACHE = STAGE1_DIR / "trials"
ENGAGED_THRESHOLD = 0.8
MIN_SESSIONS_PER_SUBJECT_CAP = 2  # a handful of very well-sampled subjects must not dominate


def select_stage1_sessions(n_sessions: int, min_biased_trials: int, seed: int = 0) -> pd.DataFrame:
    """Diverse pilot set: best session(s) per subject, ranked by biased-block
    trial count, capped so no subject contributes more than
    `MIN_SESSIONS_PER_SUBJECT_CAP` sessions (that cap is small on purpose --
    the point of a "dozens of mice" pilot is subject diversity, not depth on a
    few animals, which is what the KS023 notebook already gave us).
    """
    insertions = pd.read_parquet(config.SURVEY_DIR / "insertions.parquet")
    trial_counts = pd.read_csv(config.SURVEY_DIR / "trial_counts.csv")

    sessions = insertions[insertions["included"]][
        ["eid", "subject", "lab", "task_protocol"]].drop_duplicates("eid")
    merged = sessions.merge(trial_counts, on="eid", how="inner")
    merged = merged[merged["n_trials_biased_block"] >= min_biased_trials]
    if merged.empty:
        raise RuntimeError(
            f"no session has >= {min_biased_trials} biased-block trials; "
            "lower --min-biased-trials or re-run run_survey first")

    merged = merged.sort_values("n_trials_biased_block", ascending=False)
    rng = np.random.default_rng(seed)
    subject_order = rng.permutation(merged["subject"].unique())

    chosen: list[pd.Series] = []
    per_subject: dict[str, int] = {}
    for subject in subject_order:
        if len(chosen) >= n_sessions:
            break
        rows = merged[merged["subject"] == subject]
        for _, row in rows.iterrows():
            if per_subject.get(subject, 0) >= MIN_SESSIONS_PER_SUBJECT_CAP:
                break
            chosen.append(row)
            per_subject[subject] = per_subject.get(subject, 0) + 1
            if len(chosen) >= n_sessions:
                break
    return pd.DataFrame(chosen).reset_index(drop=True)


def cached_stage1_trials(one: Any, eid: str, task_protocol: str) -> pd.DataFrame | None:
    TRIALS_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = TRIALS_CACHE / f"{eid}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    try:
        trials = extract_session_behavior(one, eid, task_protocol)
    except TrialExtractionError:
        return None
    trials.to_parquet(cache_file, index=False)
    return trials


def load_pilot_sessions(one: Any, pilot: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sessions: dict[str, pd.DataFrame] = {}
    failed = 0
    for i, row in pilot.iterrows():
        trials = cached_stage1_trials(one, row["eid"], row.get("task_protocol", ""))
        if trials is None or trials.empty:
            failed += 1
            continue
        sessions[row["eid"]] = trials
        print(f"  [{i + 1}/{len(pilot)}] {row['eid'][:8]} ({row['subject']}): {len(trials)} trials", flush=True)
    print(f"loaded {len(sessions)}/{len(pilot)} sessions ({failed} failed)", flush=True)

    if sessions:
        # Visible early-warning for exactly the class of bug this pipeline hit
        # once already: a session silently classified "other" contributes zero
        # rows to Model 1's biased/ephys filter, and with it gone quietly the
        # only symptom downstream is "0 trials, 0 sessions" with no obvious cause.
        stage_counts = pd.concat(sessions.values())["stage"].value_counts()
        print(f"  stage distribution across loaded sessions: {stage_counts.to_dict()}", flush=True)
        if stage_counts.get("other", 0) == stage_counts.sum():
            print("  WARNING: every loaded trial classified as stage='other' -- "
                  "task_protocol likely missing from the pilot selection", flush=True)
    return sessions


def run_model1(sessions: dict[str, pd.DataFrame]) -> dict[str, Any]:
    pooled = pd.concat(sessions.values(), ignore_index=True)
    X, y, groups = history_glm_design(pooled)
    print(f"Model 1: {len(X)} trials, {groups.nunique()} sessions after NaN/stage filtering", flush=True)

    cv = fit_history_glm_nested_cv(X, y, groups)
    interp = fit_history_glm_interpretable(X, y, groups)
    return {"cv_folds": cv, "interpretable_summary": interp.summary().as_text(),
            "interpretable_params": interp.params.to_dict(),
            "interpretable_pvalues": interp.pvalues.to_dict()}


def run_glm_hmm(sessions: dict[str, pd.DataFrame]) -> dict[str, Any]:
    pooled = pd.concat(sessions.values(), ignore_index=True)
    pooled = pooled.dropna(subset=["signed_contrast", "choice_right"]).reset_index(drop=True)
    fit = fit_glm_hmm(pooled, n_states=3)
    engaged_state = label_engaged_state(fit["weights"])
    pooled["engaged_posterior"] = fit["gammas"][:, engaged_state]
    engaged_frac = float((pooled["engaged_posterior"] >= ENGAGED_THRESHOLD).mean())
    print(f"GLM-HMM: engaged state={engaged_state}, weights={fit['weights'][engaged_state]}, "
          f"{engaged_frac:.1%} of trials pass posterior>={ENGAGED_THRESHOLD}", flush=True)
    return {"weights": fit["weights"].tolist(), "trans": fit["trans"].tolist(),
            "init": fit["init"].tolist(),
            "engaged_state": engaged_state, "engaged_frac": engaged_frac,
            "log_likelihood": fit["log_likelihood"], "n_iter": fit["n_iter"],
            "trial_engaged_posterior": pooled[["eid", "engaged_posterior"]]}


def run_model2(sessions: dict[str, pd.DataFrame], engaged_posterior: pd.DataFrame | None = None) -> pd.DataFrame:
    if engaged_posterior is not None:
        by_eid = dict(tuple(engaged_posterior.groupby("eid")))
        filtered = {}
        for eid, trials in sessions.items():
            if eid not in by_eid:
                continue
            post = by_eid[eid]["engaged_posterior"].to_numpy()
            if len(post) != len(trials):
                continue  # session-count mismatch (should not happen); skip rather than misalign
            keep = trials.reset_index(drop=True)[post >= ENGAGED_THRESHOLD]
            if len(keep) >= 40:
                filtered[eid] = keep
        sessions = filtered
        print(f"Model 2 (engaged-only): {len(sessions)} sessions retain enough engaged trials", flush=True)

    return compare_belief_models(sessions)


def summarise_model2(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    return results.groupby("model").agg(
        n_sessions=("eid", "nunique"),
        mean_test_ll=("test_ll_per_trial", "mean"),
        median_test_ll=("test_ll_per_trial", "median"),
        mean_bic=("bic", "mean"),
    ).reset_index().sort_values("mean_test_ll", ascending=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sessions", type=int, default=80)
    parser.add_argument("--min-biased-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", choices=("all", "model1"), default="all",
                        help="model1 skips GLM-HMM and belief models")
    args = parser.parse_args(argv)

    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    one = get_one()

    print(f"[1/6] selecting up to {args.n_sessions} sessions "
          f"(>= {args.min_biased_trials} biased-block trials, "
          f"<= {MIN_SESSIONS_PER_SUBJECT_CAP} per subject)...", flush=True)
    pilot = select_stage1_sessions(args.n_sessions, args.min_biased_trials, seed=args.seed)
    pilot.to_csv(STAGE1_DIR / "pilot_sessions.csv", index=False)
    print(f"      selected {len(pilot)} sessions, {pilot['subject'].nunique()} subjects, "
          f"{pilot['lab'].nunique()} labs", flush=True)

    print("[2/6] loading trials (cached where available)...", flush=True)
    sessions = load_pilot_sessions(one, pilot)
    if len(sessions) < 10:
        print("too few sessions loaded to fit anything meaningful; aborting", flush=True)
        return 1

    print("[3/6] Model 1: history-dependent logistic regression...", flush=True)
    model1 = run_model1(sessions)
    model1["cv_folds"].to_csv(STAGE1_DIR / "model1_cv_folds.csv", index=False)
    (STAGE1_DIR / "model1_summary.txt").write_text(model1["interpretable_summary"])
    # Saved separately from the text summary because plots.py (and anything
    # else that wants the numbers rather than a human-readable page) needs
    # params/pvalues as data, not parsed back out of statsmodels' text table.
    coef_table = pd.DataFrame({
        "param": pd.Series(model1["interpretable_params"]),
        "pvalue": pd.Series(model1["interpretable_pvalues"]),
    })
    coef_table.to_csv(STAGE1_DIR / "model1_coefficients.csv")
    print(model1["cv_folds"][["test_ll_per_trial", "test_accuracy", "best_C"]].describe().to_string(), flush=True)

    if args.only == "model1":
        print("stopping after Model 1 (--only model1)", flush=True)
        return 0

    print("[4/6] GLM-HMM (3-state engaged/biased gate)...", flush=True)
    hmm = run_glm_hmm(sessions)
    hmm["trial_engaged_posterior"].to_parquet(STAGE1_DIR / "glm_hmm_posteriors.parquet", index=False)
    (STAGE1_DIR / "glm_hmm_summary.json").write_text(json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v)
         for k, v in hmm.items() if k != "trial_engaged_posterior"}, indent=1))

    print("[5/6] Model 2: belief-update models, all trials vs. engaged-only...", flush=True)
    all_trials_results = run_model2(sessions)
    all_trials_results.to_csv(STAGE1_DIR / "model2_all_trials.csv", index=False)
    engaged_results = run_model2(sessions, hmm["trial_engaged_posterior"])
    engaged_results.to_csv(STAGE1_DIR / "model2_engaged_only.csv", index=False)

    all_summary = summarise_model2(all_trials_results)
    engaged_summary = summarise_model2(engaged_results)
    all_summary.to_csv(STAGE1_DIR / "model2_summary_all_trials.csv", index=False)
    engaged_summary.to_csv(STAGE1_DIR / "model2_summary_engaged.csv", index=False)

    print("\n=== Model 2 summary (all trials) ===")
    print(all_summary.to_string(index=False))
    print("\n=== Model 2 summary (engaged-state trials only) ===")
    print(engaged_summary.to_string(index=False))

    print("\n[6/6] generating figures...", flush=True)
    from .plots import generate_all_figures
    figure_paths = generate_all_figures(
        model1["interpretable_params"], model1["interpretable_pvalues"],
        np.array(hmm["weights"]), hmm["engaged_state"], hmm["engaged_frac"],
        all_summary, engaged_summary)
    for p in figure_paths:
        print("      wrote", p, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
