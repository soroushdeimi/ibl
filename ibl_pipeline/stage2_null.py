"""Stage 2 (v2): decode the block prior and test it against a pseudo-session null.

This replaces the target, the cross-validation scheme and the significance
test used by `decode.py`, for reasons that were measured on this project's
own data rather than assumed. Running `decode.py`'s scheme produced no
positive R^2 in any region; a diagnostic pass showed why, and each of the
three changes below fixes one confirmed cause:

  1. TARGET is the ground-truth block `probabilityLeft`, not the fitted
     Bayes-optimal belief pi_hat. pi_hat is a derived quantity whose per-
     session hazard MLE landed on its 0.5 bound in roughly half of sessions,
     making it degenerate exactly where the analysis needed it. The block is
     the latent the animal is actually inferring, and it is measured, not
     fitted. Under interleaved CV the block decoded above chance in 95% of
     all 110 (session, region) pairs; pi_hat could be scored in only 84 of
     them, and reached a median R^2 of just 0.02 even there.

  2. TRIALS are all of them, not the GLM-HMM engaged subset. Engaged-only
     kept ~40% of trials and, worse, kept them as scattered runs, which
     interacts badly with contiguous folds. Engagement remains a Stage 1
     result; it is not a good filter to impose before decoding.

  3. FOLDS are interleaved, and significance comes from the pseudo-session
     null rather than from comparing R^2 to zero. `decode.py` used contiguous
     time-block folds to stop a decoder interpolating between temporally
     adjacent train and test trials -- a real concern, but the cure was worse
     than the disease: scoring contiguous held-out blocks against a drifting
     target gave a "predict the training mean" baseline of R^2 = -0.54, so
     every number was measured against a floor far below zero and the real
     decoder could not be told apart from it. Interleaved folds do leak, and
     session drift does help (trial index alone decodes at R^2 = 0.64) --
     but the null shares the neural data, the folds, and the block
     autocorrelation, so it prices both effects in. See `pseudosessions`.

The reported p-value is therefore "could a block sequence unrelated to this
activity have scored this well", which is the question worth answering, and
not "is R^2 > 0", which here it never is for an honest reason.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .decode import SESSIONS_DIR, load_session
from .pseudosessions import generate_pseudo_targets

STAGE2_DIR = config.SURVEY_DIR / "stage2"
ALPHAS = tuple(np.logspace(-2, 5, 15))
N_PSEUDO = 200
N_FOLDS = 5
MIN_TRIALS = 150   # a session must outlast the 90-trial pL=0.5 opening block
                   # by enough to contain real 0.2/0.8 blocks to decode
MIN_UNITS = 3
CV_SEED = 0        # real and pseudo targets MUST share folds; see score_targets


def _ridge_gcv_multitarget(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray,
                           alphas: tuple[float, ...]) -> np.ndarray:
    """Ridge for many targets at once, each choosing its own alpha by GCV.

    One SVD of the standardised training design serves every target and every
    alpha, which is what makes a 200-pseudo-session null affordable. The point
    is not only speed: fitting the pseudo-targets with the alpha chosen for the
    *real* target would be the cheap shortcut, and it would bias the null
    downward -- pseudo-targets would be denied the tuning the real target got,
    the null distribution would sit too low, and every p-value would come out
    too small. Each column of Y picks its own alpha here, so real and pseudo
    are scored by an identical procedure.

    Returns predictions for X_te, shape (n_test, n_targets).
    """
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)   # a unit silent across the whole training
                                         # fold contributes a constant column; leave
                                         # it at zero rather than dividing by ~0
    Xtr_s = (X_tr - mu) / sd
    Xte_s = (X_te - mu) / sd

    y_mean = Y_tr.mean(axis=0)
    Yc = Y_tr - y_mean

    U, s, Vt = np.linalg.svd(Xtr_s, full_matrices=False)
    s2 = s ** 2
    UtY = U.T @ Yc                        # (k, n_targets)
    n = Xtr_s.shape[0]

    # ||Yc||^2 splits into the part U spans and the part it cannot reach; the
    # unreachable part is common to every alpha, so it is computed once.
    ss_total = (Yc ** 2).sum(axis=0)
    ss_in_span = (UtY ** 2).sum(axis=0)
    ss_orthogonal = np.maximum(ss_total - ss_in_span, 0.0)

    n_targets = Yc.shape[1]
    best_gcv = np.full(n_targets, np.inf)
    best_alpha_idx = np.zeros(n_targets, dtype=int)
    for i, alpha in enumerate(alphas):
        shrink = s2 / (s2 + alpha)                     # (k,)
        resid_sq = ss_orthogonal + (((1.0 - shrink)[:, None] * UtY) ** 2).sum(axis=0)
        denom = (1.0 - shrink.sum() / n) ** 2
        gcv = resid_sq / (n * max(denom, 1e-12))
        better = gcv < best_gcv
        best_gcv[better] = gcv[better]
        best_alpha_idx[better] = i

    preds = np.empty((Xte_s.shape[0], n_targets))
    for i, alpha in enumerate(alphas):
        cols = np.flatnonzero(best_alpha_idx == i)
        if cols.size == 0:
            continue
        coef = Vt.T @ ((s / (s2 + alpha))[:, None] * UtY[:, cols])
        preds[:, cols] = Xte_s @ coef + y_mean[cols]
    return preds


def interleaved_folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Shuffled n_folds partition of trial indices, reproducible from `seed`."""
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return np.array_split(idx, n_folds)


def score_targets(X: np.ndarray, Y: np.ndarray, n_folds: int = N_FOLDS,
                  seed: int = CV_SEED) -> np.ndarray:
    """Pooled cross-validated R^2 for each column of Y, on shared folds.

    Every target -- the real block sequence and all pseudo-sessions -- is
    scored on the *same* fold partition of the *same* neural data. That is
    what lets the null absorb fold-specific luck instead of adding its own.
    Predictions are pooled across folds and scored once against the global
    mean, rather than averaging per-fold R^2, which would let a single
    low-variance fold dominate.
    """
    preds = np.empty_like(Y, dtype=float)
    all_idx = np.arange(len(X))
    for test_idx in interleaved_folds(len(X), n_folds, seed):
        train_idx = np.setdiff1d(all_idx, test_idx)
        preds[test_idx] = _ridge_gcv_multitarget(X[train_idx], Y[train_idx], X[test_idx], ALPHAS)
    ss_res = ((Y - preds) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def decode_session_region(session_data: dict[str, Any], region: str,
                          n_pseudo: int = N_PSEUDO, seed: int = 0) -> dict[str, Any] | None:
    """One (session, region): real block R^2, its pseudo-session null, and p.

    Returns None when the pair has too few units or trials to score -- an
    expected outcome for thin insertions, reported in aggregate by the caller
    rather than raised.
    """
    trials, clusters, counts = session_data["trials"], session_data["clusters"], session_data["counts"]
    region_mask = (clusters["target_region"] == region).to_numpy()
    if region_mask.sum() < MIN_UNITS:
        return None

    pL = trials["probabilityLeft"].to_numpy(dtype=float)
    valid = np.isfinite(counts[:, region_mask]).all(axis=1) & np.isfinite(pL)
    if valid.sum() < MIN_TRIALS:
        return None

    X = counts[np.ix_(valid, region_mask)]
    y_real = pL[valid]
    if y_real.std() < 1e-6:
        return None

    rng = np.random.default_rng(seed)
    # Pseudo-sessions are generated at the session's FULL trial count and then
    # masked down, not generated at the post-mask count: block structure is a
    # property of the session's trial order, and dropped trials (bad counts)
    # should punch the same holes in the null that they punch in the real target.
    pseudo_full = generate_pseudo_targets(len(pL), n_pseudo, rng)
    Y_pseudo = pseudo_full[:, valid].T
    Y_pseudo = Y_pseudo[:, Y_pseudo.std(axis=0) > 1e-6]

    Y = np.column_stack([y_real, Y_pseudo])
    scores = score_targets(X, Y)
    r2_real, r2_null = float(scores[0]), scores[1:]

    # (1 + #ties-or-better) / (1 + n) -- the standard finite-sample correction,
    # so a p-value is never reported as exactly 0 on a finite null.
    p_value = float((1 + (r2_null >= r2_real).sum()) / (1 + len(r2_null)))

    return {
        "eid": session_data["eid"], "region": region,
        "subject": session_data["manifest"].get("subject"),
        "lab": session_data["manifest"].get("lab"),
        "n_trials": int(valid.sum()), "n_units": int(region_mask.sum()),
        "r2_real": r2_real,
        "r2_null_median": float(np.median(r2_null)),
        "r2_null_p95": float(np.percentile(r2_null, 95)),
        "r2_excess": r2_real - float(np.median(r2_null)),
        "n_pseudo": int(len(r2_null)),
        "p_value": p_value,
    }


def run(eids: list[str] | None = None, n_pseudo: int = N_PSEUDO,
        verbose: bool = True) -> pd.DataFrame:
    if eids is None:
        eids = sorted(p.name for p in SESSIONS_DIR.iterdir() if p.is_dir()) if SESSIONS_DIR.exists() else []

    rows: list[dict[str, Any]] = []
    skipped = 0
    for i, eid in enumerate(eids):
        session_data = load_session(eid)
        if session_data is None or "clusters" not in session_data:
            skipped += 1
            continue
        for region in sorted(session_data["clusters"]["target_region"].unique()):
            res = decode_session_region(session_data, region, n_pseudo=n_pseudo)
            if res is None:
                continue
            rows.append(res)
            if verbose:
                print(f"[{i+1}/{len(eids)}] {eid[:8]} {res['region']:<6} "
                      f"n={res['n_trials']:4d} u={res['n_units']:3d} "
                      f"R2={res['r2_real']:+.4f} null={res['r2_null_median']:+.4f} "
                      f"p={res['p_value']:.4f}", flush=True)
    if verbose:
        print(f"sessions without usable neural data: {skipped}", flush=True)
    return pd.DataFrame(rows)


def summarise_by_region(results: pd.DataFrame) -> pd.DataFrame:
    """Region-level summary with a Fisher-combined p across that region's sessions.

    Fisher's method treats the per-session p-values as independent, which they
    are not exactly -- a region can contribute several sessions from the same
    animal. `n_subjects` is reported next to `n_sessions` so a region whose
    evidence rests on one heavily-recorded mouse is visible as such rather
    than being read as many independent replications. The combined p is also
    conservative for a second reason: each session's p is discrete with a
    floor of 1/(n_pseudo+1), so no session can contribute more evidence than
    the null's resolution allows.
    """
    if results.empty:
        return results

    def _fisher(p: pd.Series) -> float:
        return float(stats.combine_pvalues(p.to_numpy(), method="fisher")[1])

    summary = results.groupby("region").agg(
        n_sessions=("eid", "nunique"),
        n_subjects=("subject", "nunique"),
        n_units_median=("n_units", "median"),
        n_trials_median=("n_trials", "median"),
        r2_real_median=("r2_real", "median"),
        r2_null_median=("r2_null_median", "median"),
        r2_excess_median=("r2_excess", "median"),
        n_sig_sessions=("p_value", lambda p: int((p < 0.05).sum())),
        fisher_p=("p_value", _fisher),
    ).reset_index()
    summary["frac_sig"] = summary["n_sig_sessions"] / summary["n_sessions"]
    summary = summary.sort_values("fisher_p").reset_index(drop=True)

    # Every region in `config.TARGET_REGIONS` is tested, so the smallest of
    # eight p-values must not be read as if it were the only test run.
    # Benjamini-Hochberg is the primary correction (these regions are a
    # screen, and controlling false discovery rate is the right trade here);
    # Bonferroni is reported alongside as the conservative bound.
    m = len(summary)
    ranks = np.arange(1, m + 1)
    raw = summary["fisher_p"].to_numpy()
    summary["bonferroni_p"] = np.minimum(raw * m, 1.0)
    # Enforce monotonicity from the largest p downward, as BH requires --
    # without it a region can show a smaller q than one ranked above it.
    summary["bh_q"] = np.minimum.accumulate((raw * m / ranks)[::-1])[::-1].clip(max=1.0)
    return summary


def main() -> int:
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    results = run()
    if results.empty:
        print("no (session, region) pair had enough data to decode")
        return 1
    results.to_csv(STAGE2_DIR / "block_null_results.csv", index=False)
    summary = summarise_by_region(results)
    summary.to_csv(STAGE2_DIR / "block_null_summary_by_region.csv", index=False)
    print("\n" + summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
