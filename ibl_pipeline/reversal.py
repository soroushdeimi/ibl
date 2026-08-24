"""Stage 3: can the prior be dissociated from choice history at a block reversal?

This exists because Stage 2's claim did not survive its own controls.
`pseudopopulation.py` showed that decoding the block from pre-stimulus activity
is fully explained by recent choice and reward history: the behaviour-only
decoder reaches AUC 0.985-1.000 in every region, beating the pseudo-session
null with no neural data at all. Block and behavioural history are collinear in
steady state, so no statistic separates them there.

THE DISSOCIATION AND ITS LIMITS. Immediately after a 0.2 <-> 0.8 reversal the
two stop being collinear and become opposed: the new block is one thing, the
animal's last choices still reflect the old one. History is not removed from
that window -- its sign is flipped -- so a classifier trained *inside* the
window would just learn the flipped mapping. What works is the transfer: train
on steady-state trials, where prior and history agree, then test after a
reversal, where they disagree. The SIGN of the transfer says which variable the
decoder was using: above 0.5 it tracks the new block, below 0.5 the old one.

The behaviour-only decoder validates the design, and it passes: AUC 0.096-0.165
at lag 1 in all seven regions, crossing 0.5 by lag 2.

WHAT LAG 1 CANNOT SHOW -- read this before interpreting any number here. The
analysis window is 400 ms *before* the stimulus of the first new-block trial.
At that moment no information about the new block has reached the animal at
all: not the stimulus, not an outcome, not a choice. So finding that
pre-stimulus activity still reports the old block at lag 1 is entailed by the
structure of the task, not discovered in the brain. Lag 1 is where the
dissociation is clean, which makes it the right place to *validate* the design
and the wrong place to measure how fast a prior updates. Measuring update
latency needs lags >= 2, where the animal has seen evidence -- but there the
confound is already back (behaviour-only is above 0.65 from lag 2 and above
0.78 from lag 3), so only the `residuals` curve is interpretable, and only
with per-lag confidence intervals. Those are computed here for exactly that
reason: the raw CP residual curve reads 0.44, 0.44, 0.47, 0.41, 0.57, ... and
without intervals that is noise a reader will mistake for a rise.

THE NULL'S REMAINING ASYMMETRY. `_pseudo_labels` randomises the block sequence
but leaves the real behavioural history sitting in the real neural data. The
real labels can therefore reach the old-block signal through history coding;
the fake labels cannot. An AUC below 0.5 on real data is thus partly
structural rather than a finding -- the same defect that sank Stage 2, with
its sign flipped. The evidence that this is not hypothetical: both significant
effects vanish once history is regressed out (CP 0.361 -> 0.440, p 0.005 ->
0.264; MOp 0.279 -> 0.438, p 0.010 -> 0.517). Treat `residuals` as the claim
and `counts` as an upper bound, never the reverse.

WHAT THIS MODULE ACTUALLY ESTABLISHES. Not that the neural prior updates
slowly. Rather: in this task prior and choice history are inseparable almost
everywhere, and the single window that does separate them is the one window
where no new evidence has yet arrived. That is a property of the experimental
design, not of the analysis, and it means this dataset cannot answer the
proposal's second question as posed.

WHY NOT A PSEUDO-POPULATION. Sessions carry ~11 reversals, so a single lag
offers 2-4 trials per class per session. Pooling units would add features, not
trials, and it is what amplified the history confound in Stage 2 (12 sessions
turned a 74%-accurate confound into AUC 1.000). Each session gets its own
decoder; the *predictions* are pooled after within-session standardisation.

PRE-REGISTERED DESIGN, and one documented change:

  * reversal      a change between pL = 0.2 and pL = 0.8. The opening
                  0.5 -> biased transition is not a reversal: no opposite prior
                  precedes it, so there is no dissociation to exploit.
  * lag           1 = first trial carrying the new block value.
  * primary test  lag 1 only. Registered as lags 1-5 and narrowed after the
                  behaviour-only validation curve -- not after any neural
                  result -- showed lags 2+ do not dissociate: the animal's
                  choice follows the stimulus, and stimulus side is drawn from
                  the new block immediately, so history re-acquires the new
                  block within a trial or two without any belief having
                  changed. Lags 1-5 are still reported as `auc_primary5` and
                  are confounded.
  * time course   lags 1-15, per lag, with bootstrap intervals over reversals.
  * steady state  lag >= 10, training only.
  * directions    pooled. They CANNOT be a covariate: inside the reversal
                  window the direction of a reversal and the identity of the
                  new block are the same variable, so a "direction regressor"
                  would be the target. Each direction's mean decoder output is
                  reported per lag instead, which is the symmetry check the
                  covariate was meant to provide.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .pseudopopulation import collect_region_sessions
from .pseudosessions import generate_block_sequence
from .stage2_null import ALPHAS, _ridge_gcv_multitarget

STAGE3_DIR = config.SURVEY_DIR / "stage3"
PRIMARY_LAGS = (1,)
SECONDARY_LAGS = (1, 2, 3, 4, 5)
TIME_COURSE_LAGS = tuple(range(1, 16))
STEADY_MIN_LAG = 10
MIN_STEADY_PER_CLASS = 30
MIN_TEST_TRIALS = 20
N_PSEUDO = 200
N_BOOTSTRAP = 2000
FEATURES = ("counts", "residuals", "confounds")


def reversal_structure(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lag, reversal ordinal) per trial, both -1 before the first reversal.

    Lag is 1 on the first trial carrying the new block value. The ordinal
    identifies which reversal a trial belongs to, which is the unit the
    bootstrap resamples -- trials within one reversal share an animal state and
    are not independent, so resampling trials would understate the intervals.
    NaN labels (the pL = 0.5 opening block) neither break the count nor count
    as a reversal, since a block change is defined only between the two biased
    values.
    """
    lags = np.full(len(labels), -1, dtype=int)
    ordinals = np.full(len(labels), -1, dtype=int)
    previous = None
    reversal_at = None
    ordinal = -1
    for i, value in enumerate(labels):
        if np.isnan(value):
            continue
        if previous is not None and value != previous:
            reversal_at = i
            ordinal += 1
        previous = value
        if reversal_at is not None:
            lags[i] = i - reversal_at + 1
            ordinals[i] = ordinal
    return lags, ordinals


def _fit_and_transfer(X: np.ndarray, y: np.ndarray, steady: np.ndarray,
                      test: np.ndarray) -> np.ndarray | None:
    """Train on `steady`, predict on `test`, standardised by the training
    predictions so sessions can be pooled on a common scale. None when the
    training set is too small or one-sided to define a decoder."""
    if test.sum() == 0 or steady.sum() < 2 * MIN_STEADY_PER_CLASS:
        return None
    if min((y[steady] == c).sum() for c in (0.0, 1.0)) < MIN_STEADY_PER_CLASS:
        return None

    both = np.flatnonzero(steady | test)
    train_idx = np.flatnonzero(steady)
    predictions = _ridge_gcv_multitarget(X[train_idx], y[train_idx][:, None], X[both], ALPHAS)[:, 0]

    position = {j: k for k, j in enumerate(both)}
    train_pred = predictions[[position[j] for j in train_idx]]
    if train_pred.std() < 1e-12:
        return None

    out = np.full(len(y), np.nan)
    out[both] = (predictions - train_pred.mean()) / train_pred.std()
    return out


def transfer_predictions(sessions: list[dict[str, Any]], label_sets: list[np.ndarray],
                         feature: str) -> pd.DataFrame:
    """Every post-reversal trial's standardised decoder output, pooled across
    sessions, tagged with its lag and its reversal identity."""
    rows = []
    for s_idx, (session, labels) in enumerate(zip(sessions, label_sets)):
        lags, ordinals = reversal_structure(labels)
        y = np.nan_to_num(labels, nan=-1.0)
        usable = session["valid"] & np.isfinite(labels)
        steady = usable & (lags >= STEADY_MIN_LAG)
        test = usable & np.isin(lags, TIME_COURSE_LAGS)

        X = session[feature] if feature != "counts" else session["counts"]
        standardised = _fit_and_transfer(X, y, steady, test)
        if standardised is None:
            continue
        for j in np.flatnonzero(test):
            if np.isfinite(standardised[j]):
                rows.append((int(lags[j]), float(standardised[j]), float(y[j]),
                             f"{s_idx}:{ordinals[j]}"))
    return pd.DataFrame(rows, columns=["lag", "pred", "truth", "reversal"])


def curve_from_predictions(predictions: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Pooled AUC per lag. Scored once on the pooled predictions rather than
    averaged over per-session AUCs: at 2-4 trials per class per session a
    per-session AUC is mostly noise, while the pooled ranking is well defined."""
    curve = {}
    for lag, group in predictions.groupby("lag"):
        if len(group) < MIN_TEST_TRIALS or group["truth"].nunique() < 2:
            continue
        curve[int(lag)] = {
            "auc": float(roc_auc_score(group["truth"], group["pred"])),
            "n": int(len(group)),
            "n_reversals": int(group["reversal"].nunique()),
            "mean_pred_to_08": float(group.loc[group["truth"] == 1.0, "pred"].mean()),
            "mean_pred_to_02": float(group.loc[group["truth"] == 0.0, "pred"].mean()),
        }
    return curve


def bootstrap_ci(predictions: pd.DataFrame, n_boot: int = N_BOOTSTRAP,
                 seed: int = 0) -> dict[int, tuple[float, float]]:
    """Percentile bootstrap over REVERSALS, not trials.

    Trials from one reversal share the animal's state at that moment and are
    plainly not independent; resampling them individually would produce
    intervals far too tight and invite exactly the "read a trend into noise"
    error this curve is prone to.
    """
    rng = np.random.default_rng(seed)
    units = predictions["reversal"].unique()
    by_unit = {u: g for u, g in predictions.groupby("reversal")}

    samples: dict[int, list[float]] = {k: [] for k in TIME_COURSE_LAGS}
    for _ in range(n_boot):
        drawn = rng.choice(units, size=len(units), replace=True)
        resampled = pd.concat([by_unit[u] for u in drawn], ignore_index=True)
        for lag, group in resampled.groupby("lag"):
            if len(group) < MIN_TEST_TRIALS or group["truth"].nunique() < 2:
                continue
            samples[int(lag)].append(roc_auc_score(group["truth"], group["pred"]))

    return {lag: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for lag, v in samples.items() if len(v) >= n_boot // 10}


def weighted_auc(curve: dict[int, dict[str, float]], lags: tuple[int, ...]) -> float:
    entries = [(curve[k]["auc"], curve[k]["n"]) for k in lags if k in curve]
    if not entries:
        return float("nan")
    aucs = np.array([e[0] for e in entries])
    weights = np.array([e[1] for e in entries], dtype=float)
    return float((aucs * weights).sum() / weights.sum())


def _pseudo_labels(session: dict[str, Any], rng: np.random.Generator,
                   max_retries: int = 50) -> np.ndarray:
    """Fake block labelling with enough steady-state trials in both classes."""
    from .pseudopopulation import block_labels
    for _ in range(max_retries):
        labels = block_labels(generate_block_sequence(session["n_trials"], rng))
        lags, _ = reversal_structure(labels)
        steady = session["valid"] & np.isfinite(labels) & (lags >= STEADY_MIN_LAG)
        if min((steady & (labels == c)).sum() for c in (0.0, 1.0)) >= MIN_STEADY_PER_CLASS:
            return labels
    return labels


def _two_sided_p(real: float, null: np.ndarray) -> float:
    """Distance-from-null-centre p. Two-sided because the SIGN of the transfer
    is the finding: a decoder locked onto the OLD block is as informative as
    one locked onto the new, and must not be scored as "no effect"."""
    if not null.size or not np.isfinite(real):
        return float("nan")
    centre = float(np.median(null))
    return float((1 + (np.abs(null - centre) >= abs(real - centre)).sum()) / (1 + null.size))


def test_region(region: str, n_pseudo: int = N_PSEUDO, seed: int = 0,
                verbose: bool = True) -> tuple[dict[str, Any], pd.DataFrame] | None:
    sessions = collect_region_sessions(region)
    if len(sessions) < 2:
        if verbose:
            print(f"{region:<6} skipped: {len(sessions)} contributing session(s)", flush=True)
        return None

    real_labels = [s["labels"] for s in sessions]
    real_predictions = {f: transfer_predictions(sessions, real_labels, f) for f in FEATURES}
    real_curves = {f: curve_from_predictions(real_predictions[f]) for f in FEATURES}
    intervals = {f: bootstrap_ci(real_predictions[f]) for f in FEATURES}

    rng = np.random.default_rng(seed)
    null_primary: dict[str, list[float]] = {f: [] for f in FEATURES}
    null_per_lag: dict[str, dict[int, list[float]]] = {f: {k: [] for k in TIME_COURSE_LAGS}
                                                       for f in FEATURES}
    for _ in range(n_pseudo):
        fake = [_pseudo_labels(s, rng) for s in sessions]
        for f in FEATURES:
            curve = curve_from_predictions(transfer_predictions(sessions, fake, f))
            null_primary[f].append(weighted_auc(curve, PRIMARY_LAGS))
            for lag, values in curve.items():
                null_per_lag[f][lag].append(values["auc"])

    result: dict[str, Any] = {
        "region": region,
        "n_sessions": len(sessions),
        "n_subjects": len({s["subject"] for s in sessions}),
        "n_units": int(sum(s["n_units"] for s in sessions)),
        "n_pseudo": int(n_pseudo),
        "underpowered": bool(len(sessions) < config.MIN_SESSIONS_FOR_POWER
                             or len({s["subject"] for s in sessions}) < config.MIN_SUBJECTS_FOR_POWER),
    }
    for f in FEATURES:
        suffix = "" if f == "counts" else f"_{f}"
        null = np.array([v for v in null_primary[f] if np.isfinite(v)])
        real = weighted_auc(real_curves[f], PRIMARY_LAGS)
        result[f"auc{suffix}"] = real
        result[f"auc_primary5{suffix}"] = weighted_auc(real_curves[f], SECONDARY_LAGS)
        result[f"auc_null_median{suffix}"] = float(np.median(null)) if null.size else float("nan")
        result[f"p_value{suffix}"] = _two_sided_p(real, null)

    rows = []
    for f in FEATURES:
        for lag, values in sorted(real_curves[f].items()):
            low, high = intervals[f].get(lag, (np.nan, np.nan))
            null = np.array(null_per_lag[f][lag])
            rows.append({"region": region, "feature": f, "lag": lag, **values,
                         "ci_low": low, "ci_high": high,
                         "null_median": float(np.median(null)) if null.size else np.nan,
                         "p_value": _two_sided_p(values["auc"], null)})

    if verbose:
        print(f"{region:<6} sess={result['n_sessions']:3d} units={result['n_units']:4d} | "
              f"lag1 neural {result['auc']:.3f} (p={result['p_value']:.3f}) | "
              f"resid {result['auc_residuals']:.3f} (p={result['p_value_residuals']:.3f}) | "
              f"behav {result['auc_confounds']:.3f} (p={result['p_value_confounds']:.3f})", flush=True)
    return result, pd.DataFrame(rows)


def run(regions: tuple[str, ...] = config.TARGET_REGIONS, n_pseudo: int = N_PSEUDO,
        verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = [r for r in (test_region(region, n_pseudo=n_pseudo, verbose=verbose)
                           for region in regions) if r is not None]
    if not outputs:
        return pd.DataFrame(), pd.DataFrame()

    summary = pd.DataFrame([o[0] for o in outputs]).sort_values("p_value").reset_index(drop=True)
    m = len(summary)
    raw = summary["p_value"].to_numpy()
    summary["bh_q"] = np.minimum.accumulate(
        (raw * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1.0)
    return summary, pd.concat([o[1] for o in outputs], ignore_index=True)


def main() -> int:
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    summary, curves = run()
    if summary.empty:
        print("no region had two or more contributing sessions")
        return 1
    summary.to_csv(STAGE3_DIR / "reversal_summary.csv", index=False)
    curves.to_csv(STAGE3_DIR / "reversal_time_course.csv", index=False)

    columns = ["region", "n_sessions", "n_subjects", "n_units", "underpowered",
               "auc", "auc_null_median", "p_value", "bh_q",
               "auc_residuals", "p_value_residuals", "auc_confounds", "p_value_confounds"]
    print("\n" + summary[columns].to_string(index=False))
    print("\nvalidation: `auc_confounds` must sit BELOW 0.5 at lag 1. If it does not, the "
          "reversal window has not dissociated prior from history.\nreminder: lag 1 carries "
          "no evidence about the new block, so an old-block reading there is entailed by the "
          "task, not discovered; and the null cannot reach the old-block signal that history "
          "gives the real labels, so read `residuals`, not `counts`.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
