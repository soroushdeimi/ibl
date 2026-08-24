"""Stage 2 (v1, SUPERSEDED by `stage2_null.py`): decode Stage 1's winning
behavioural model from population activity (proposal Q8, steps 6 and 9).

SUPERSEDED, and kept only because the reasoning below is worth preserving.
This module reported negative R^2 in every region and read that as "no
prior signal". It was not: `stage2_controls.py` showed the extraction is
sound (trial index decodes at R^2 = 0.64, the ground-truth block above
chance in 95% of session/region pairs), and three choices here were
destroying the effect --

  * the target, a fitted pi_hat whose hazard MLE sat on its 0.5 bound in
    about half of sessions;
  * engaged-only trials, which cut N by ~60% and left scattered runs;
  * scoring contiguous time-block folds against zero, when the "predict
    the training mean" baseline on those same folds is itself R^2 = -0.54,
    so zero was never the right reference point.

`stage2_null.py` fixes all three and gets significance from the
pseudo-session null (step 7) that this module deferred. Use that instead.
Nothing below is wrong as code; the analysis it implements is.

For each extracted session (`data/sessions/<eid>/`, written by `extract.py`)
and each target region it has QC-passing units in:

1. Recompute the GLM-HMM engaged-state posterior for *this session's own*
   trial sequence, using Stage 1's already-fitted weights/transition matrix
   (`data/survey/stage1/glm_hmm_summary.json`) -- not by trying to re-align
   Stage 1's saved per-trial posteriors, which were computed on a pooled,
   separately-NaN-filtered dataframe and are not guaranteed to line up
   row-for-row with `extract.py`'s own trials table. Recomputing from the
   saved model parameters is cheap (no re-fitting) and removes that alignment
   risk entirely.
2. Reconstruct the trial-by-trial Bayes-optimal prior pi_hat using that
   session's own fitted hazard rate, pulled from Stage 1's per-session model
   comparison (`model2_engaged_only.csv`) -- the model Stage 1 found wins
   specifically on engaged trials.
3. Restrict to engaged trials (posterior >= ENGAGED_THRESHOLD) with both
   valid neural data and a valid pi_hat.
4. Ridge regression decoding pi_hat from the region's pre-stimulus population
   spike counts, with nested cross-validation using *contiguous time-block*
   folds (not shuffled trials) -- the prior is slowly autocorrelated, so a
   shuffled split would let a decoder "cheat" by interpolating between
   temporally adjacent train and test trials rather than actually reading the
   prior out of activity.
5. A same-fold motor-only competing decoder (wheel/whisker/pupil covariates
   from `extract.py`'s `covariates.parquet`) -- if the neural decoder does not
   beat this, a "prior representation" claim is not supported (proposal step 9).

Step 7 (pseudo-session null testing) was deferred here on the grounds that
this module's ridge/CV machinery should be trusted first. That was the right
instinct and the wrong order: the null is what *establishes* whether the
machinery works, because without it there is no reference point to judge an
R^2 against. It now lives in `pseudosessions.py` / `stage2_null.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .behavior_models import _forward_backward, _state_logistic_prob, bayes_optimal_belief

SESSIONS_DIR = config.SESSIONS_DIR
STAGE1_DIR = config.SURVEY_DIR / "stage1"
STAGE2_DIR = config.SURVEY_DIR / "stage2"
ENGAGED_THRESHOLD = 0.8
MOTOR_COVARIATES = ["wheel_speed_mean", "wheel_velocity_mean", "whisker_me_mean", "pupil_diameter_mean"]
MIN_TRIALS_FOR_DECODING = 60  # below this, a 5x5 nested CV has near-empty folds
MIN_UNITS_FOR_DECODING = 3
MIN_PI_HAT_STD = 0.01  # below this, pi_hat is degenerate (see decode_session_region)


def load_glm_hmm_params() -> dict[str, Any]:
    return json.loads((STAGE1_DIR / "glm_hmm_summary.json").read_text())


def load_bayes_optimal_hazards() -> dict[str, float]:
    """Per-session fitted hazard rate from Stage 1's engaged-only bayes_optimal
    fit -- the winning model. `params` was saved as a JSON-encoded list
    `[hazard, gamma, eta, lapse]`; only the belief parameter (hazard) is
    needed here, the policy parameters (gamma/eta/lapse) were only ever for
    scoring choice likelihood in Stage 1, not for reconstructing pi_hat.
    """
    eng = pd.read_csv(STAGE1_DIR / "model2_engaged_only.csv")
    bo = eng[eng["model"] == "bayes_optimal"].copy()
    bo["hazard"] = bo["params"].apply(lambda s: json.loads(s)[0] if isinstance(s, str) else s[0])
    return dict(zip(bo["eid"], bo["hazard"]))


def session_engaged_gamma(trials: pd.DataFrame, hmm: dict[str, Any]) -> np.ndarray:
    """Engaged-state posterior for one session's own trial sequence, computed
    from Stage 1's already-fitted GLM-HMM (weights/trans/init) via a single
    forward-backward pass -- no re-fitting, no cross-session alignment.
    """
    weights = np.array(hmm["weights"])
    trans = np.array(hmm["trans"])
    init = np.array(hmm["init"])
    engaged_state = hmm["engaged_state"]

    contrast = trials["signed_contrast"].to_numpy(dtype=float)
    y = trials["choice_right"].to_numpy(dtype=float)
    n_states = weights.shape[0]
    emis = np.stack([
        np.where(y == 1, _state_logistic_prob(contrast, weights[k]),
                  1 - _state_logistic_prob(contrast, weights[k]))
        for k in range(n_states)
    ], axis=1)
    emis = np.clip(emis, 1e-6, 1.0)
    gamma, _, _ = _forward_backward(emis, trans, init)
    return gamma[:, engaged_state]


def load_session(eid: str) -> dict[str, Any] | None:
    """All of one extracted session's saved outputs, or None if the manifest
    says it never got usable units (a legitimate outcome, not an error)."""
    session_dir = SESSIONS_DIR / eid
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") not in ("complete", "complete_no_units"):
        return None
    trials_path = session_dir / "trials.parquet"
    if not trials_path.exists():
        return None
    trials = pd.read_parquet(trials_path)
    result: dict[str, Any] = {"eid": eid, "manifest": manifest, "trials": trials}

    clusters_path = session_dir / "clusters.parquet"
    counts_path = session_dir / "prestim_counts.npy"
    if clusters_path.exists() and counts_path.exists():
        result["clusters"] = pd.read_parquet(clusters_path)
        result["counts"] = np.load(counts_path)
    covariates_path = session_dir / "covariates.parquet"
    if covariates_path.exists():
        result["covariates"] = pd.read_parquet(covariates_path)
    return result


def time_block_folds(n: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous, non-overlapping, order-preserving folds -- each fold's test
    block is a single contiguous run of trials, never an interleaved sample,
    so a fold boundary is the only place train and test trials are adjacent
    in time (see module docstring for why that matters here).
    """
    edges = np.linspace(0, n, n_splits + 1).astype(int)
    folds = []
    all_idx = np.arange(n)
    for i in range(n_splits):
        test_idx = all_idx[edges[i]:edges[i + 1]]
        train_idx = np.concatenate([all_idx[:edges[i]], all_idx[edges[i + 1]:]])
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def _ridge_r2_nested_cv(X: np.ndarray, y: np.ndarray, alphas: tuple[float, ...],
                         n_outer: int = 5, n_inner: int = 5,
                         min_fold_std: float = 0.05) -> list[dict[str, Any]]:
    """Ridge regression, nested time-block CV, R^2 per outer fold.

    A test fold whose y is (near-)constant is skipped rather than scored: R^2
    = 1 - SS_res/SS_tot is undefined in the limit SS_tot -> 0, and any
    nonzero residual there blows up into an arbitrarily large negative
    number that then dominates a mean over folds -- found by actually seeing
    a -1.29e8 fold in real output (the belief can saturate near a bound for
    an entire contiguous block when engaged trials cluster during a long,
    one-sided run), not anticipated up front. `MIN_PI_HAT_STD` in
    decode_session_region catches this at the whole-session level; this is
    the same check at fold granularity, which a session can fail locally even
    when its full engaged-trial set has enough overall variance to pass that
    coarser gate. Threshold is 0.05, not the smaller 0.01 first tried: a real
    fold at std=0.0108 (barely over 0.01) still produced R^2=-351.7 -- pi_hat
    lives in roughly [0.2, 0.8], so 0.05 is about 8% of that span and is the
    value that actually stopped producing three- and four-digit-negative
    folds against this project's real sessions, not a value chosen a priori.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rows = []
    for train_idx, test_idx in time_block_folds(len(y), n_outer):
        if y[test_idx].std() < min_fold_std:
            continue
        best_alpha, best_score = alphas[0], -np.inf
        for alpha in alphas:
            inner_scores = []
            for tr2, va2 in time_block_folds(len(train_idx), n_inner):
                tr_abs, va_abs = train_idx[tr2], train_idx[va2]
                if y[va_abs].std() < min_fold_std:
                    continue
                xs = StandardScaler().fit(X[tr_abs])
                model = Ridge(alpha=alpha).fit(xs.transform(X[tr_abs]), y[tr_abs])
                inner_scores.append(model.score(xs.transform(X[va_abs]), y[va_abs]))
            mean_score = float(np.mean(inner_scores)) if inner_scores else -np.inf
            if mean_score > best_score:
                best_score, best_alpha = mean_score, alpha
        if best_score == -np.inf:
            continue  # every inner fold was degenerate too; alpha never chosen on real signal

        xs = StandardScaler().fit(X[train_idx])
        model = Ridge(alpha=best_alpha).fit(xs.transform(X[train_idx]), y[train_idx])
        r2 = model.score(xs.transform(X[test_idx]), y[test_idx])
        rows.append({"alpha": best_alpha, "r2": r2, "n_train": len(train_idx), "n_test": len(test_idx)})
    return rows


def decode_session_region(session_data: dict[str, Any], region: str, hmm: dict[str, Any],
                           hazard: float, alphas: tuple[float, ...] = (0.1, 1, 10, 100, 1000)
                           ) -> dict[str, Any] | None:
    """One (session, region): builds engaged-trial pi_hat and the region's
    spike-count design matrix, then runs the neural and motor-only decoders.
    Returns None (not an error) when there is not enough usable data --
    logged by the caller's aggregate summary, not raised session by session.
    """
    trials = session_data["trials"]
    if "clusters" not in session_data:
        return None
    clusters, counts = session_data["clusters"], session_data["counts"]

    region_mask = (clusters["target_region"] == region).to_numpy()
    if region_mask.sum() < MIN_UNITS_FOR_DECODING:
        return None

    gamma = session_engaged_gamma(trials, hmm)
    stim_right = trials["contrastRight"].notna().to_numpy(dtype=float)
    pi_hat = bayes_optimal_belief(stim_right, np.array([hazard]))

    engaged = gamma >= ENGAGED_THRESHOLD
    valid_counts = np.isfinite(counts[:, region_mask]).all(axis=1)
    keep = engaged & valid_counts & np.isfinite(pi_hat)
    if keep.sum() < MIN_TRIALS_FOR_DECODING:
        return None

    X_neural = counts[np.ix_(keep, region_mask)]
    y = pi_hat[keep]

    if y.std() < MIN_PI_HAT_STD:
        # A session whose per-session hazard MLE landed on the parameter's own
        # upper bound (0.5) gives a pi_hat that is exactly constant -- proven
        # in sanity_check.check_bayes_optimal_belief_synthetic, hazard=0.5
        # destroys all information every trial. Decoding a constant target is
        # meaningless, and sklearn's R^2 special-cases zero-variance y_true to
        # 1.0 for an exact (here: trivial) match -- which would silently read
        # as "perfect decoding" rather than the degenerate fit it actually is.
        # Found by inspecting a real R^2=1.0 result, not anticipated in advance.
        return None

    neural_folds = _ridge_r2_nested_cv(X_neural, y, alphas)
    if not neural_folds:
        # Every outer fold was locally degenerate even though the session's
        # overall pi_hat variance passed MIN_PI_HAT_STD -- e.g. engaged trials
        # cluster into one long, one-sided run per fold. Nothing usable to
        # report; not the same as "no signal" (a real, scoreable R^2 below 0).
        return None

    result: dict[str, Any] = {
        "eid": session_data["eid"], "region": region,
        "n_trials": int(keep.sum()), "n_units": int(region_mask.sum()),
        "n_folds_used": len(neural_folds),
        "neural_r2_mean": float(np.mean([f["r2"] for f in neural_folds])),
        "neural_r2_folds": [f["r2"] for f in neural_folds],
    }

    covariates = session_data.get("covariates")
    if covariates is not None:
        motor = covariates[MOTOR_COVARIATES].to_numpy(dtype=float)[keep]
        motor_valid = np.isfinite(motor).all(axis=1)
        if motor_valid.sum() >= MIN_TRIALS_FOR_DECODING and y[motor_valid].std() >= MIN_PI_HAT_STD:
            motor_folds = _ridge_r2_nested_cv(motor[motor_valid], y[motor_valid], alphas)
            result["motor_r2_mean"] = float(np.mean([f["r2"] for f in motor_folds])) if motor_folds else None
            result["motor_n_trials"] = int(motor_valid.sum())
        else:
            result["motor_r2_mean"] = None
    else:
        result["motor_r2_mean"] = None

    return result


def run_stage2_decoding(eids: list[str] | None = None) -> pd.DataFrame:
    hmm = load_glm_hmm_params()
    hazards = load_bayes_optimal_hazards()

    if eids is None:
        eids = [p.name for p in SESSIONS_DIR.iterdir() if p.is_dir()] if SESSIONS_DIR.exists() else []

    rows = []
    skipped = {"no_manifest": 0, "no_hazard": 0, "no_region_data": 0}
    for eid in eids:
        session_data = load_session(eid)
        if session_data is None:
            skipped["no_manifest"] += 1
            continue
        if eid not in hazards:
            skipped["no_hazard"] += 1
            continue
        if "clusters" not in session_data:
            skipped["no_region_data"] += 1
            continue

        for region in sorted(session_data["clusters"]["target_region"].unique()):
            res = decode_session_region(session_data, region, hmm, hazards[eid])
            if res is not None:
                rows.append(res)

    print(f"skipped: {skipped}", flush=True)
    return pd.DataFrame(rows)


def summarise_by_region(results: pd.DataFrame) -> pd.DataFrame:
    """Region-level summary, sorted and led by the *median* R^2, not the mean.

    R^2 has no lower bound, and even after the fold-level degenerate-target
    guard, a handful of sessions can still land a badly-generalizing fold
    (small held-out N, imperfect regularisation) with a large negative value.
    The median is what a reader should treat as "the" number per region; the
    mean is kept alongside for transparency but will swing on exactly the
    kind of outlier session the median is robust to -- report both, but do
    not let the mean's ordering stand in for the region ranking.
    """
    if results.empty:
        return results
    return results.groupby("region").agg(
        n_sessions=("eid", "nunique"),
        n_units_median=("n_units", "median"),
        n_trials_median=("n_trials", "median"),
        neural_r2_median=("neural_r2_mean", "median"),
        neural_r2_mean=("neural_r2_mean", "mean"),
        motor_r2_median=("motor_r2_mean", "median"),
        motor_r2_mean=("motor_r2_mean", "mean"),
    ).reset_index().sort_values("neural_r2_median", ascending=False)


def main() -> int:
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    results = run_stage2_decoding()
    if results.empty:
        print("no (session, region) pairs had enough data to decode; "
              "is extract.py still running / has it finished any sessions yet?")
        return 1
    results.to_csv(STAGE2_DIR / "decode_results.csv", index=False)
    summary = summarise_by_region(results)
    summary.to_csv(STAGE2_DIR / "decode_summary_by_region.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
