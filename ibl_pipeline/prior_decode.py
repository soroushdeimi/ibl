"""Stage 2 (v4): decode the Bayes-optimal prior, on Findling et al.'s own protocol.

Everything before this decoded the raw block `probabilityLeft`. Findling,
Hubert et al. (Nature 2025, s41586-025-09226-1) decode the *Bayes-optimal
prior* instead, and state explicitly that decoding the true block prior gives
weaker results. Until this project uses the same target, a reader can dismiss
the disagreement -- they find the prior in 20-30% of regions, we find it in
none -- as a difference of target rather than a difference of control. This
module closes that escape route by matching their protocol where we can, and
naming the one place we cannot.

MATCHED. Target is the Bayes-optimal prior: the predictive P(stimulus on the
right) entering each trial, from an exact forward filter over the block state
(`behavior_models.bayes_optimal_belief`). Decoding is per session with
interleaved cross-validation and R^2, and p-values are combined per region with
Fisher's method -- their procedure, not the pseudo-population, which in this
project amplified the history confound rather than the signal.

The hazard rate comes from the TRUE generative process, not from a per-session
fit. Mean block length measured on this dataset is 50 trials, so h = 1/50 =
0.02. This is a deliberate change from `decode.py`, whose per-session hazard
MLE landed on its 0.5 bound in about half of sessions and produced a degenerate
target there. A fixed known hazard has no per-session freedom to overfit, and
it is what a normative observer of this task would actually use. It is an
approximation in one respect worth stating: real block lengths are a truncated
exponential on [20, 100], so the true process is not memoryless and a constant
hazard slightly misstates the reversal probability just after a block starts.

MATCHED -- the analysis window. Findling et al. use -600 to -100 ms before
stimulus onset (`config.PRESTIM_WINDOW_S`, extract schema 2).

THE NULL, AND WHY IT IS BUILT THIS WAY. A belief cannot be nulled by shuffling
it: it is a deterministic function of the stimulus sequence. So, as in their
paper, each null draw regenerates a block sequence from the rig's generative
process, draws a counterfactual stimulus sequence from it, and recomputes the
Bayes-optimal prior from that. Because the design matrix is fixed per session,
the real target and all counterfactuals are scored together on shared folds.

WHAT THIS MODULE ADDS THAT THE PAPER DOES NOT. Their null regenerates the
stimulus sequence but does not simulate choices, so the animal's real choice
history stays in the neural data while the counterfactual prior is unrelated to
it -- their significance test therefore does not separate prior coding from
past-action coding. They found this dependence themselves, by orthogonalisation
(the influence of past actions "extends at least to the past five trials") and
in a correct-vs-error asymmetry that matches an action-kernel model rather than
a Bayes-optimal one. This module puts that check inside the test rather than
beside it, by running the identical pipeline on three feature sets:

    counts      pre-stimulus spike counts, the headline decoder
    residuals   the same counts with choice/reward history regressed out
    confounds   that history alone, with no neural data whatsoever

`confounds` is the decisive control: whatever it scores is available without a
brain. Read `residuals` as the claim and `counts` as an upper bound.

Note what a negative result here does and does not license. It supports: "in
these regions, with a -600 to -100 ms window and a linear decoder, no prior
signal separable from action history was found." It does not support "the
Findling result is an artefact" -- a handful of regions against 242, one
decoder class, and a control the paper itself reports in another form.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .behavior_models import bayes_optimal_belief
from .decode import SESSIONS_DIR, load_session
from .pseudopopulation import _residualise
from .pseudosessions import generate_block_sequence, generate_stimulus_sides
from .stage2_null import score_targets

STAGE2_DIR = config.SURVEY_DIR / "stage2"
TRUE_HAZARD = 0.02            # 1 / (measured mean block length of 50 trials)
N_PSEUDO = 200
MIN_TRIALS = 150
MIN_UNITS = 3
FEATURES = ("counts", "residuals", "confounds")
CONFOUND_COLUMNS = ("prev_choice_right", "prev_rewarded",
                    "prev2_choice_right", "prev2_rewarded")


def bayes_optimal_prior(stim_right: np.ndarray, hazard: float = TRUE_HAZARD) -> np.ndarray:
    return bayes_optimal_belief(stim_right, np.array([hazard]))


def counterfactual_priors(n_trials: int, n_pseudo: int, rng: np.random.Generator,
                          hazard: float = TRUE_HAZARD) -> np.ndarray:
    """`n_pseudo` x `n_trials` counterfactual Bayes-optimal priors.

    Block sequence -> stimulus sequence -> belief, the same three steps that
    produced the real target, so the counterfactual inherits the real one's
    autocorrelation and its range without inheriting any relationship to the
    recorded neurons. Degenerate draws (no variance) are rejected: R^2 against
    a constant target is undefined.
    """
    out = np.empty((n_pseudo, n_trials))
    filled = 0
    while filled < n_pseudo:
        blocks = generate_block_sequence(n_trials, rng)
        candidate = bayes_optimal_prior(generate_stimulus_sides(blocks, rng), hazard)
        if candidate.std() < 1e-6:
            continue
        out[filled] = candidate
        filled += 1
    return out


def session_features(session_data: dict[str, Any], region: str
                     ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray] | None:
    """(feature matrices, target, valid mask) for one (session, region)."""
    trials, clusters, counts = session_data["trials"], session_data["clusters"], session_data["counts"]
    region_mask = (clusters["target_region"] == region).to_numpy()
    if region_mask.sum() < MIN_UNITS:
        return None

    region_counts = counts[:, region_mask]
    stim_right = trials["contrastRight"].notna().to_numpy(dtype=float)
    prior = bayes_optimal_prior(stim_right)

    confounds = np.column_stack([trials[c].to_numpy(dtype=float) for c in CONFOUND_COLUMNS])
    confounds = np.column_stack([confounds, confounds[:, 0] * confounds[:, 1]])

    valid = (np.isfinite(region_counts).all(axis=1) & np.isfinite(prior)
             & np.isfinite(confounds).all(axis=1))
    if valid.sum() < MIN_TRIALS or prior[valid].std() < 1e-6:
        return None

    features = {
        "counts": region_counts,
        "residuals": _residualise(region_counts, confounds, valid),
        "confounds": confounds,
    }
    return features, prior, valid


def decode_session_region(session_data: dict[str, Any], region: str,
                          n_pseudo: int = N_PSEUDO, seed: int = 0) -> dict[str, Any] | None:
    prepared = session_features(session_data, region)
    if prepared is None:
        return None
    features, prior, valid = prepared

    rng = np.random.default_rng(seed)
    null_priors = counterfactual_priors(len(prior), n_pseudo, rng)[:, valid].T
    null_priors = null_priors[:, null_priors.std(axis=0) > 1e-6]
    Y = np.column_stack([prior[valid], null_priors])

    result: dict[str, Any] = {
        "eid": session_data["eid"], "region": region,
        "subject": session_data["manifest"].get("subject"),
        "n_trials": int(valid.sum()),
        "n_units": int(features["counts"].shape[1]),
    }
    for feature in FEATURES:
        # X is fixed for this (session, region, feature), so the real target and
        # every counterfactual are solved together on one shared SVD per fold.
        scores = score_targets(features[feature][valid], Y)
        real, null = float(scores[0]), scores[1:]
        suffix = "" if feature == "counts" else f"_{feature}"
        result[f"r2{suffix}"] = real
        result[f"r2_null_median{suffix}"] = float(np.median(null))
        result[f"r2_excess{suffix}"] = real - float(np.median(null))
        result[f"p_value{suffix}"] = float((1 + (null >= real).sum()) / (1 + len(null)))
    return result


def run(eids: list[str] | None = None, n_pseudo: int = N_PSEUDO,
        verbose: bool = True) -> pd.DataFrame:
    if eids is None:
        eids = sorted(p.name for p in SESSIONS_DIR.iterdir() if p.is_dir()) if SESSIONS_DIR.exists() else []

    rows = []
    for i, eid in enumerate(eids):
        session_data = load_session(eid)
        if session_data is None or "clusters" not in session_data:
            continue
        for region in sorted(session_data["clusters"]["target_region"].unique()):
            result = decode_session_region(session_data, region, n_pseudo=n_pseudo)
            if result is None:
                continue
            rows.append(result)
            if verbose:
                print(f"[{i+1}/{len(eids)}] {eid[:8]} {result['region']:<6} "
                      f"n={result['n_trials']:4d} u={result['n_units']:3d} | "
                      f"neural {result['r2']:+.4f} p={result['p_value']:.3f} | "
                      f"resid {result['r2_residuals']:+.4f} p={result['p_value_residuals']:.3f} | "
                      f"behav {result['r2_confounds']:+.4f} p={result['p_value_confounds']:.3f}",
                      flush=True)
    return pd.DataFrame(rows)


def summarise_by_region(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results

    def _fisher(p: pd.Series) -> float:
        return float(stats.combine_pvalues(p.to_numpy(), method="fisher")[1])

    aggregations: dict[str, Any] = {
        "n_sessions": ("eid", "nunique"),
        "n_subjects": ("subject", "nunique"),
        "n_units_median": ("n_units", "median"),
    }
    for feature in FEATURES:
        suffix = "" if feature == "counts" else f"_{feature}"
        aggregations[f"r2_median{suffix}"] = (f"r2{suffix}", "median")
        aggregations[f"r2_excess_median{suffix}"] = (f"r2_excess{suffix}", "median")
        aggregations[f"n_sig{suffix}"] = (f"p_value{suffix}", lambda p: int((p < 0.05).sum()))
        aggregations[f"fisher_p{suffix}"] = (f"p_value{suffix}", _fisher)

    summary = results.groupby("region").agg(**aggregations).reset_index()
    summary["underpowered"] = ((summary["n_sessions"] < config.MIN_SESSIONS_FOR_POWER)
                               | (summary["n_subjects"] < config.MIN_SUBJECTS_FOR_POWER))
    summary = summary.sort_values("fisher_p").reset_index(drop=True)

    m = len(summary)
    for feature in FEATURES:
        suffix = "" if feature == "counts" else f"_{feature}"
        raw = summary[f"fisher_p{suffix}"].to_numpy()
        order = np.argsort(raw)
        ranked = raw[order]
        q = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1.0)
        out = np.empty(m)
        out[order] = q
        summary[f"bh_q{suffix}"] = out
    return summary


def main() -> int:
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    results = run()
    if results.empty:
        print("no (session, region) pair had enough data")
        return 1
    results.to_csv(STAGE2_DIR / "prior_decode_results.csv", index=False)
    summary = summarise_by_region(results)
    summary.to_csv(STAGE2_DIR / "prior_decode_summary_by_region.csv", index=False)

    columns = ["region", "n_sessions", "n_subjects", "underpowered",
               "r2_excess_median", "fisher_p", "bh_q",
               "r2_excess_median_residuals", "fisher_p_residuals", "bh_q_residuals",
               "r2_excess_median_confounds", "fisher_p_confounds"]
    print("\n" + summary[columns].to_string(index=False))
    print(f"\ntarget: Bayes-optimal prior, hazard {TRUE_HAZARD} from the measured "
          "mean block length; window -600 to -100 ms (Findling et al.).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
