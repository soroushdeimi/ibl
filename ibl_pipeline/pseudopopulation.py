"""Stage 2 (v3): pool units across sessions into a per-region pseudo-population.

`stage2_null.py` decodes one session at a time, and that is where it runs out
of road: the median (session, region) pair has 10-51 simultaneously recorded
units, and no region survived FDR correction. Pooling units across every
session that recorded a region gets CP from 51 units to 770, CA1 to 499 and
MOs to 426 -- the regime where the IBL brain-wide map finds prior coding at
all.

WHAT A PSEUDO-POPULATION COSTS. Units pooled this way were never recorded
together, so pseudo-trial i pairs session A's trial with an unrelated trial
from session B. Noise correlations between the pooled units are destroyed by
construction. A pseudo-population therefore answers "is the block prior
recoverable from this region's units, given enough of them", NOT "does the
population as recorded represent the prior". If pooling turns a null result
positive, that is evidence about coverage, not about a mechanism the
single-session analysis missed.

CONSTRUCTION. Only the biased blocks (pL = 0.2 and 0.8) are used; the 90-trial
pL = 0.5 opening block is neither class. For each region, every session with
enough units and enough trials per block contributes `n_per_cond` trials of
each block, sampled without replacement. Because the decoder treats trials as
exchangeable, the *order* of pseudo-trials carries no information -- only
which real trials were assigned to which class matters.

WHY THE NULL MUST REBUILD THE POPULATION. The obvious null -- keep the built
population and permute the labels -- is wrong here, and quantifiably so. Block
identity is confounded with time-within-session: blocks are contiguous, firing
rates drift (trial index decodes at R^2 = 0.64 in this dataset), so a session's
pL = 0.8 trials differ from its pL = 0.2 trials for reasons that have nothing
to do with the prior. Permuting labels destroys that confound and gives a null
centred at chance, which makes the confound look like signal. So each null
draw instead re-labels every session with a *fresh block sequence from the
rig's own generative process* (`pseudosessions`) and rebuilds the population
from scratch. The fake blocks are contiguous in time exactly like real ones,
so the drift confound is reproduced; they are causally unrelated to the
neurons, so real prior coding is not. Both nulls are computed and reported --
`p_value` from the pseudo-session null is the one to believe over
`p_value_naive`, which is included so the size of the drift confound is
visible rather than merely asserted.

WHAT THE PSEUDO-SESSION NULL STILL DOES NOT CONTROL. It fixes drift. It does
not fix behavioural history, and that gap is large enough to overturn a
result. The block is heavily confounded with what the animal just did: in
pL = 0.2 blocks the previous choice was rightward on 71.5% of trials, in
pL = 0.8 blocks on 23.3%, so previous choice alone identifies the block about
74% of the time. The pre-stimulus window sits immediately after that movement
and its reward. Crucially the null cannot absorb this, because it randomises
the blocks while leaving the real choice sequence in the neural data intact:
the real labels can exploit history coding, the fake labels cannot.

So `feature="confounds"` runs the entire pipeline on the behavioural history
and NO neural data. On MOp that scores AUC 0.985 against a null of 0.604 --
the pseudo-session null is beaten decisively without a single spike. Any
region where that control is significant cannot have its neural result read
as prior coding. `feature="residuals"` is the corrected analysis: the same
units after choice/reward history is regressed out of them. Read
`p_value_residuals` as the claim, `p_value` as the upper bound, and
`p_value_confounds` as the check on whether the two can differ at all.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from . import config
from .decode import SESSIONS_DIR, load_session
from .pseudosessions import generate_block_sequence
from .stage2_null import ALPHAS, _ridge_gcv_multitarget

STAGE2_DIR = config.SURVEY_DIR / "stage2"
MIN_UNITS = 3
MIN_TRIALS_PER_COND = 100   # to be included, a session needs this many usable
                            # trials in BOTH pL=0.2 and pL=0.8
MAX_TRIALS_PER_COND = 150   # cap, so one unusually long session cannot force
                            # every other session to sample with replacement
N_PSEUDO = 200
N_RESAMPLE = 3              # pseudo-population draws averaged per score; applied
                            # identically to the real and null scores (see score_once)
N_FOLDS = 5
MAX_LABEL_RETRIES = 50


def block_labels(pL: np.ndarray) -> np.ndarray:
    """pL -> {0.0, 1.0, nan}, nan for the pL = 0.5 opening block.

    `probabilityLeft` is the probability the STIMULUS appears on the left, so
    pL = 0.8 is a *left*-biased block (verified on this dataset: 80.4% of its
    trials have a left stimulus) and pL = 0.2 is right-biased. Class 1 here is
    pL = 0.8, i.e. left-biased. The mapping direction does not affect AUC --
    flipping both the labels and the fitted decoder leaves it unchanged -- but
    it does decide how the sign of any coefficient must be read.
    """
    rounded = np.round(pL.astype(float), 2)
    labels = np.full(len(rounded), np.nan)
    labels[rounded == 0.2] = 0.0
    labels[rounded == 0.8] = 1.0
    return labels


def _residualise(counts: np.ndarray, confounds: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Counts with the linear contribution of recent choice/reward history removed.

    Fitted once per session on that session's own valid trials, by ordinary
    least squares with an intercept. This is *not* refitted inside the
    cross-validation loop, and deliberately so: the regressors are behavioural
    history only, never the block label, so nothing about the target leaks
    across folds. What it does do is remove block-related variance wherever
    block and recent choice are collinear -- which in this task is most of it.
    That makes the residualised analysis strictly conservative: it can only
    understate prior coding, never manufacture it.
    """
    design = np.column_stack([np.ones(valid.sum()), confounds[valid]])
    coefficients, *_ = np.linalg.lstsq(design, counts[valid], rcond=None)
    residuals = counts.astype(float).copy()
    residuals[valid] = counts[valid] - design @ coefficients
    return residuals


def collect_region_sessions(region: str) -> list[dict[str, Any]]:
    """Every session contributing usable units in `region`, with its counts
    matrix already restricted to that region and to trials with finite counts."""
    contributors = []
    for path in sorted(SESSIONS_DIR.iterdir()):
        if not path.is_dir():
            continue
        session_data = load_session(path.name)
        if session_data is None or "clusters" not in session_data:
            continue
        region_mask = (session_data["clusters"]["target_region"] == region).to_numpy()
        if region_mask.sum() < MIN_UNITS:
            continue

        counts = session_data["counts"][:, region_mask]
        pL = session_data["trials"]["probabilityLeft"].to_numpy(dtype=float)
        valid = np.isfinite(counts).all(axis=1) & np.isfinite(pL)
        labels = block_labels(pL)
        if min((valid & (labels == c)).sum() for c in (0.0, 1.0)) < MIN_TRIALS_PER_COND:
            continue

        trials = session_data["trials"]
        confounds = np.column_stack([
            trials["prev_choice_right"].to_numpy(dtype=float),
            trials["prev_rewarded"].to_numpy(dtype=float),
            trials["prev2_choice_right"].to_numpy(dtype=float),
            trials["prev2_rewarded"].to_numpy(dtype=float),
        ])
        confounds = np.column_stack([confounds, confounds[:, 0] * confounds[:, 1]])
        # The first two trials have no history and would otherwise poison both
        # the residualisation and the behaviour-only control.
        valid = valid & np.isfinite(confounds).all(axis=1)
        if min((valid & (labels == c)).sum() for c in (0.0, 1.0)) < MIN_TRIALS_PER_COND:
            continue

        contributors.append({
            "eid": path.name,
            "subject": session_data["manifest"].get("subject"),
            "counts": counts,
            "confounds": confounds,
            "residuals": _residualise(counts, confounds, valid),
            "valid": valid,
            "labels": labels,
            "n_trials": len(pL),
            "n_units": int(region_mask.sum()),
        })
    return contributors


def pseudo_labels_for(session: dict[str, Any], n_per_cond: int,
                      rng: np.random.Generator) -> np.ndarray:
    """A fake block labelling of one session, drawn from the rig's generative
    process and rejected until both classes have enough usable trials.

    Retries are bounded; on exhaustion the last draw is returned and the caller
    falls back to sampling with replacement for that session. That fallback is
    applied to the null only, which makes the null slightly *noisier* rather
    than slightly easier to beat -- the safe direction for a significance test.
    """
    for _ in range(MAX_LABEL_RETRIES):
        labels = block_labels(generate_block_sequence(session["n_trials"], rng))
        if min((session["valid"] & (labels == c)).sum() for c in (0.0, 1.0)) >= n_per_cond:
            return labels
    return labels


def build_pseudopopulation(sessions: list[dict[str, Any]], label_sets: list[np.ndarray],
                           n_per_cond: int, rng: np.random.Generator,
                           feature: str = "counts") -> tuple[np.ndarray, np.ndarray]:
    """Stack `n_per_cond` trials of each class from every session, side by side
    in feature space. Returns (X, y) with X shape (2*n_per_cond, total_features).

    `feature` selects what each session contributes:
      "counts"    raw pre-stimulus spike counts -- the headline analysis.
      "residuals" those counts with recent choice/reward history regressed out.
      "confounds" the choice/reward history itself and no neural data at all,
                  which turns the whole machinery into a test of whether the
                  null can be beaten without any brain activity.
    """
    per_session = []
    for session, labels in zip(sessions, label_sets):
        source = session[feature] if feature != "counts" else session["counts"]
        class_blocks = []
        for cls in (0.0, 1.0):
            candidates = np.flatnonzero(session["valid"] & (labels == cls))
            replace = len(candidates) < n_per_cond
            chosen = rng.choice(candidates, size=n_per_cond, replace=replace)
            class_blocks.append(source[chosen])
        per_session.append(np.vstack(class_blocks))

    X = np.hstack(per_session)
    y = np.concatenate([np.zeros(n_per_cond), np.ones(n_per_cond)])
    order = rng.permutation(len(y))   # so folds are not aligned with the class blocks
    return X[order], y[order]


def _stratified_interleaved_folds(y: np.ndarray, n_folds: int,
                                  rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffled folds that keep both classes present in every test fold --
    an all-one-class fold makes AUC undefined."""
    folds: list[list[np.ndarray]] = [[] for _ in range(n_folds)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, chunk in enumerate(np.array_split(idx, n_folds)):
            folds[i].append(chunk)
    return [np.concatenate(parts) for parts in folds]


def cross_validated_auc(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    """Pooled out-of-fold AUC from ridge predictions of the 0/1 class label.

    AUC rather than accuracy: the ridge output is continuous, and AUC uses all
    of it without needing a threshold that would itself have to be tuned.
    """
    predictions = np.empty(len(y))
    all_idx = np.arange(len(y))
    for test_idx in _stratified_interleaved_folds(y, N_FOLDS, rng):
        train_idx = np.setdiff1d(all_idx, test_idx)
        predictions[test_idx] = _ridge_gcv_multitarget(
            X[train_idx], y[train_idx][:, None], X[test_idx], ALPHAS)[:, 0]
    return float(roc_auc_score(y, predictions))


def score_once(sessions: list[dict[str, Any]], label_sets: list[np.ndarray],
               n_per_cond: int, rng: np.random.Generator,
               n_resample: int = N_RESAMPLE, feature: str = "counts") -> float:
    """Mean AUC over `n_resample` independent pseudo-population draws.

    Averaging is applied to the real score and to every null score alike. A
    real score averaged over draws and a null score from a single draw would
    not be comparable: averaging shrinks the spread, so the real value would
    be measured on a tighter distribution than the null it is judged against.
    """
    return float(np.mean([
        cross_validated_auc(*build_pseudopopulation(sessions, label_sets, n_per_cond, rng, feature), rng)
        for _ in range(n_resample)
    ]))


def test_region(region: str, n_pseudo: int = N_PSEUDO, seed: int = 0,
                verbose: bool = True) -> dict[str, Any] | None:
    sessions = collect_region_sessions(region)
    if len(sessions) < 2:
        if verbose:
            print(f"{region:<6} skipped: only {len(sessions)} contributing session(s)", flush=True)
        return None

    n_per_cond = min(MAX_TRIALS_PER_COND,
                     min(min((s["valid"] & (s["labels"] == c)).sum() for c in (0.0, 1.0))
                         for s in sessions))
    rng = np.random.default_rng(seed)
    real_labels = [s["labels"] for s in sessions]

    # Three feature sets, scored against ONE shared set of pseudo-session
    # labellings so the three tests are paired and directly comparable:
    #   counts     the headline neural decoder;
    #   residuals  the same units after recent choice/reward history is
    #              regressed out -- because in this task the block is heavily
    #              confounded with what the animal just did (in pL=0.2 blocks
    #              the previous choice was rightward on 71.5% of trials, in
    #              pL=0.8 blocks on 23.3%), and the pre-stimulus window sits
    #              immediately after that movement and its reward;
    #   confounds  that behavioural history ALONE, no neural data. This is the
    #              decisive control: the pseudo-session null randomises blocks
    #              but leaves the real choice sequence in place, so it does NOT
    #              price the confound in. If "confounds" comes out significant,
    #              a significant "counts" result cannot be read as prior coding.
    features = ("counts", "residuals", "confounds")
    auc_real = {f: score_once(sessions, real_labels, n_per_cond, rng, feature=f) for f in features}

    null_draws: dict[str, list[float]] = {f: [] for f in features}
    for _ in range(n_pseudo):
        fake_labels = [pseudo_labels_for(s, n_per_cond, rng) for s in sessions]
        for f in features:
            null_draws[f].append(score_once(sessions, fake_labels, n_per_cond, rng, feature=f))
    auc_null_all = {f: np.array(v) for f, v in null_draws.items()}
    auc_null = auc_null_all["counts"]

    # Naive null: keep the real population, permute the labels. Reported only
    # to expose the drift confound, never as the significance test -- and note
    # it is not even a like-for-like comparison: it is scored from a single
    # pseudo-population draw while `auc_real` averages `N_RESAMPLE` of them, so
    # its spread is wider than the real score's. Its *centre* is what carries
    # the message (how far above 0.5 the confound alone puts a decoder);
    # `p_value_naive` is indicative only.
    X_fixed, y_fixed = build_pseudopopulation(sessions, real_labels, n_per_cond, rng, "counts")
    auc_naive = np.array([
        cross_validated_auc(X_fixed, rng.permutation(y_fixed), rng) for _ in range(n_pseudo)
    ])

    result = {
        "region": region,
        "n_sessions": len(sessions),
        "n_subjects": len({s["subject"] for s in sessions}),
        "n_units": int(sum(s["n_units"] for s in sessions)),
        "n_per_cond": int(n_per_cond),
        "auc_naive_null_median": float(np.median(auc_naive)),
        "p_value_naive": float((1 + (auc_naive >= auc_real["counts"]).sum()) / (1 + len(auc_naive))),
        "n_pseudo": int(len(auc_null)),
        # config declares these thresholds and survey.py enforces them, but no
        # Stage 2 module was checking them -- which is how MOp came to be the
        # one "significant" region on 4 sessions from 3 subjects, below the
        # project's own stated minimum. Flagged rather than excluded: dropping
        # a region is the analyst's call, silently reporting it as powered is not.
        "underpowered": bool(len(sessions) < config.MIN_SESSIONS_FOR_POWER
                             or len({s["subject"] for s in sessions}) < config.MIN_SUBJECTS_FOR_POWER),
    }
    for f in features:
        suffix = "" if f == "counts" else f"_{f}"
        real, null = auc_real[f], auc_null_all[f]
        result[f"auc_real{suffix}"] = real
        result[f"auc_null_median{suffix}"] = float(np.median(null))
        result[f"auc_excess{suffix}"] = real - float(np.median(null))
        result[f"p_value{suffix}"] = float((1 + (null >= real).sum()) / (1 + len(null)))
    if verbose:
        print(f"{region:<6} sess={result['n_sessions']:3d} units={result['n_units']:4d} "
              f"trials={2*n_per_cond:4d} | "
              f"neural {result['auc_real']:.3f}/{result['auc_null_median']:.3f} p={result['p_value']:.4f} | "
              f"resid {result['auc_real_residuals']:.3f}/{result['auc_null_median_residuals']:.3f} "
              f"p={result['p_value_residuals']:.4f} | "
              f"behav-only {result['auc_real_confounds']:.3f}/{result['auc_null_median_confounds']:.3f} "
              f"p={result['p_value_confounds']:.4f}", flush=True)
    return result


def run(regions: tuple[str, ...] = config.TARGET_REGIONS, n_pseudo: int = N_PSEUDO,
        verbose: bool = True) -> pd.DataFrame:
    rows = [r for r in (test_region(region, n_pseudo=n_pseudo, verbose=verbose)
                        for region in regions) if r is not None]
    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results = results.sort_values("p_value").reset_index(drop=True)
    m = len(results)
    raw = results["p_value"].to_numpy()
    results["bonferroni_p"] = np.minimum(raw * m, 1.0)
    results["bh_q"] = np.minimum.accumulate(
        (raw * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1.0)
    return results


def main() -> int:
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    results = run()
    if results.empty:
        print("no region had two or more contributing sessions")
        return 1
    results.to_csv(STAGE2_DIR / "pseudopop_results.csv", index=False)
    columns = ["region", "n_sessions", "n_subjects", "n_units", "n_per_cond",
               "auc_real", "auc_null_median", "auc_excess", "p_value", "bh_q",
               "auc_real_residuals", "auc_excess_residuals", "p_value_residuals",
               "auc_real_confounds", "auc_excess_confounds", "p_value_confounds"]
    print("\n" + results[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
