"""Positive controls for the Stage 2 decoder: is a null result the data's, or ours?

Stage 2 v1 (`decode.py`) returned negative R^2 in every region. A null result
from a decoding pipeline has two very different explanations -- the signal is
absent, or the pipeline cannot see a signal that is present -- and nothing in
the result itself distinguishes them. This module distinguishes them, by
running the *same* counts matrices through the *same* ridge machinery against
targets whose decodability is already known:

  trial_index   Firing rates drift over a session, so trial index must decode
                easily. This is really an alignment assay: if the counts
                matrix rows did not correspond to the trials table rows, this
                is the target that would collapse, and it is the cheapest
                place to catch that mistake.
  prev_choice   The previous trial's choice, from pre-stimulus activity.
                An established choice-history signal; expected to be modest
                but above chance, so it calibrates "what a small real effect
                looks like here".
  block_pL      The ground-truth block (0.2 vs 0.8 trials). This is the
                latent the animal is inferring. If it does not decode, then a
                *derived belief* like pi_hat has no chance, and the problem is
                upstream of any modelling choice.
  pi_hat        Stage 2 v1's actual target, scored the same way, so the two
                are directly comparable.

Scoring is interleaved 5-fold CV, deliberately: these are controls, and the
question is only "is there any signal in these counts at all", so the goal is
maximum sensitivity, not an unbiased effect size. The numbers here are
inflated by temporal leakage and drift and are NOT significance tests -- that
is what `stage2_null.py`'s pseudo-session null is for. Read this module as a
smoke test with a pass/fail, not as a result.

Recorded outcome on the 70 usable sessions (2026-08-22): trial index R^2 =
0.64 with 99% of pairs positive, block 0.573 balanced accuracy with 95% of
pairs above chance, previous choice 0.540 with 84% above chance, pi_hat R^2 =
0.020 with 77% positive -- but pi_hat was scoreable in only 84 of the 110
pairs at all, the rest being sessions whose hazard MLE left it degenerate.
Pipeline exonerated; target and CV scheme indicted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config
from .behavior_models import bayes_optimal_belief
from .decode import (MIN_UNITS_FOR_DECODING, SESSIONS_DIR, load_bayes_optimal_hazards,
                     load_session)

STAGE2_DIR = config.SURVEY_DIR / "stage2"
ALPHAS = np.logspace(-1, 4, 8)
MIN_TRIALS = 100
N_FOLDS = 5


def r2_interleaved_cv(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Pooled out-of-fold R^2 under a shuffled split."""
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in KFold(N_FOLDS, shuffle=True, random_state=seed).split(X):
        scaler = StandardScaler().fit(X[train_idx])
        model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(X[train_idx]), y[train_idx])
        pred[test_idx] = model.predict(scaler.transform(X[test_idx]))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def balanced_accuracy_cv(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Pooled out-of-fold balanced accuracy, stratified so both classes appear
    in every fold. Balanced rather than raw accuracy because block trials are
    not evenly split within a session -- raw accuracy would reward a decoder
    for learning the majority block."""
    if len(np.unique(y)) < 2 or np.bincount(y.astype(int)).min() < N_FOLDS:
        return float("nan")
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        model = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(X[train_idx]), y[train_idx])
        pred[test_idx] = model.predict(scaler.transform(X[test_idx]))
    return float(balanced_accuracy_score(y, pred))


def controls_for_session_region(session_data: dict[str, Any], region: str,
                                hazards: dict[str, float]) -> dict[str, Any] | None:
    trials, clusters, counts = session_data["trials"], session_data["clusters"], session_data["counts"]
    region_mask = (clusters["target_region"] == region).to_numpy()
    if region_mask.sum() < MIN_UNITS_FOR_DECODING:
        return None

    usable = np.isfinite(counts[:, region_mask]).all(axis=1)
    X = counts[np.ix_(usable, region_mask)]
    t = trials[usable].reset_index(drop=True)
    if len(t) < MIN_TRIALS:
        return None

    row: dict[str, Any] = {"eid": session_data["eid"], "region": region,
                           "n_trials": len(t), "n_units": int(region_mask.sum())}

    row["r2_trial_index"] = r2_interleaved_cv(X, np.arange(len(t), dtype=float))

    prev_choice = t["prev_choice_right"].to_numpy(dtype=float)
    has_prev = np.isfinite(prev_choice)   # trial 0 has no predecessor
    row["bacc_prev_choice"] = (balanced_accuracy_cv(X[has_prev], prev_choice[has_prev])
                               if has_prev.sum() > MIN_TRIALS else np.nan)

    pL = np.round(t["probabilityLeft"].to_numpy(dtype=float), 2)
    biased = np.isin(pL, [0.2, 0.8])      # drop the 0.5 opening block: it is neither class
    row["n_block_trials"] = int(biased.sum())
    row["bacc_block"] = (balanced_accuracy_cv(X[biased], (pL[biased] == 0.8).astype(float))
                         if biased.sum() > MIN_TRIALS else np.nan)

    hazard = hazards.get(session_data["eid"])
    if hazard is not None:
        stim_right = t["contrastRight"].notna().to_numpy(dtype=float)
        pi_hat = bayes_optimal_belief(stim_right, np.array([hazard]))
        finite = np.isfinite(pi_hat)
        row["pi_hat_std"] = float(np.nanstd(pi_hat))
        row["r2_pi_hat"] = (r2_interleaved_cv(X[finite], pi_hat[finite])
                            if finite.sum() > MIN_TRIALS and np.nanstd(pi_hat) > 0.01 else np.nan)
    return row


def run(eids: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    hazards = load_bayes_optimal_hazards()
    if eids is None:
        eids = sorted(p.name for p in SESSIONS_DIR.iterdir() if p.is_dir()) if SESSIONS_DIR.exists() else []

    rows = []
    for eid in eids:
        session_data = load_session(eid)
        if session_data is None or "clusters" not in session_data:
            continue
        for region in sorted(session_data["clusters"]["target_region"].unique()):
            row = controls_for_session_region(session_data, region, hazards)
            if row is None:
                continue
            rows.append(row)
            if verbose:
                print(f"{eid[:8]} {region:<6} trial_idx={row['r2_trial_index']:+.3f} "
                      f"block={row['bacc_block']:.3f} prev_choice={row['bacc_prev_choice']:.3f}",
                      flush=True)
    return pd.DataFrame(rows)


def verdict(results: pd.DataFrame) -> str:
    """Pass/fail on the one control that is an outright bug detector.

    Only trial index gets a hard threshold. The other three are effect sizes
    whose true values are unknown, so a number that looks low there is a
    finding to think about, not a failure to assert.
    """
    median_alignment = results["r2_trial_index"].median()
    if median_alignment < 0.1:
        return (f"FAIL: trial index decodes at median R^2 = {median_alignment:.3f}. Firing rates "
                "drift, so this should be strongly positive -- suspect counts/trials misalignment "
                "in extract.py before interpreting anything else here.")
    return f"PASS: counts and trials are aligned (trial index R^2 = {median_alignment:.3f})."


def main() -> int:
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    results = run()
    if results.empty:
        print("no (session, region) pair had enough data for the controls")
        return 1
    results.to_csv(STAGE2_DIR / "controls_results.csv", index=False)

    summary = results.groupby("region").agg(
        n=("eid", "size"), n_units_median=("n_units", "median"), n_trials_median=("n_trials", "median"),
        r2_trial_index=("r2_trial_index", "median"), bacc_prev_choice=("bacc_prev_choice", "median"),
        bacc_block=("bacc_block", "median"), r2_pi_hat=("r2_pi_hat", "median"),
    ).reset_index().sort_values("bacc_block", ascending=False)
    summary.to_csv(STAGE2_DIR / "controls_summary_by_region.csv", index=False)

    print("\n" + summary.to_string(index=False))
    print("\noverall medians (chance = 0.5 for bacc_*, 0 for r2_*):")
    for column, chance in [("r2_trial_index", 0.0), ("bacc_prev_choice", 0.5),
                           ("bacc_block", 0.5), ("r2_pi_hat", 0.0)]:
        values = results[column].dropna()
        print(f"  {column:18s} {values.median():+.4f}   above chance in {(values > chance).mean():.0%} of pairs")
    print("\n" + verdict(results))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
