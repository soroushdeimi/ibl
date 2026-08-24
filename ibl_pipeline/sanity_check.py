"""Verification that the extraction agrees with what ONE reports directly.

    python -m ibl_pipeline.sanity_check [--eid ...] [--pid ...] [--offline]

The point is to catch *silent* extraction bugs -- an off-by-one in the trial
filter, a cluster/channel misalignment, a spike-count window that quietly counts
the wrong interval -- before they propagate into decoding results, where they
would look like a real (or absent) effect rather than a bug.

Offline checks use synthetic spike trains and need no network; online checks
re-derive the same numbers straight from the ONE API and compare.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config
from .behavior import add_history_features, extract_session_behavior, load_trials
from .behavior_models import _forward_backward, bayes_optimal_belief, fixed_rate_belief
from .decode import _ridge_r2_nested_cv, time_block_folds
from .spikes import load_cluster_table, load_spike_trains, prestim_spike_counts, region_atlas_ids

# A small, well-behaved public session used as the reference case. Fixed rather
# than random so a failure is reproducible.
DEFAULT_EID = "08102cfc-a040-4bcf-b63c-faa0f4914a6f"
DEFAULT_PID = "77cb0293-f054-4488-b34e-65869853e20e"


class SanityCheckError(AssertionError):
    """Raised when extracted data disagrees with the reference source."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SanityCheckError(message)


# --- offline checks -------------------------------------------------------

def check_prestim_counts_synthetic() -> str:
    """Spike counts must match a hand-computed answer on a known train.

    Boundary behaviour is asserted explicitly: both window edges are inclusive,
    and flipping either searchsorted side would shift every count by a spike.
    """
    # All times are exact binary fractions: with np.arange the accumulated 0.1
    # error puts the edge spike on the wrong side of the boundary and the test
    # measures float drift instead of the window logic.
    # Cluster 0 straddles both edges of the window [9.6, 10.0]; cluster 1 fires
    # once inside it.
    times = np.array([9.5, 9.6, 9.75, 9.875, 10.0, 10.125, 9.75])
    clusters = np.array([0, 0, 0, 0, 0, 0, 1])
    events = np.array([10.0, np.nan])

    counts = prestim_spike_counts(times, clusters, np.array([0, 1]), events, (-0.4, 0.0))
    # Both edges are inclusive (searchsorted left/right): 9.6, 9.75, 9.875, 10.0
    # count; 9.5 and 10.125 fall outside.
    _check(counts[0, 0] == 4, f"expected 4 spikes for cluster 0, got {counts[0, 0]}")
    _check(counts[0, 1] == 1, f"expected 1 spike for cluster 1, got {counts[0, 1]}")
    _check(np.isnan(counts[1, 0]), "NaN event time must yield NaN counts, not 0")

    # A window that excludes everything must give 0, not NaN: "no spikes" and
    # "no data" are different states downstream.
    empty = prestim_spike_counts(times, clusters, np.array([0]), np.array([5.0]), (-0.4, 0.0))
    _check(empty[0, 0] == 0, "empty window must count 0")
    return "prestim_spike_counts matches hand-computed counts (incl. NaN and empty cases)"


def check_history_features_synthetic() -> str:
    """Lagged history must shift within session and never across a boundary."""
    frame = pd.DataFrame({
        "eid": ["a"] * 3 + ["b"] * 3,
        "choice": [-1, 1, -1, 1, -1, 1],
        "choice_right": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "rewarded": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        "intervals_0": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    })
    out = add_history_features(frame, lags=(1, 2))
    first_of_b = out[out["eid"] == "b"].iloc[0]
    _check(np.isnan(first_of_b["prev_choice_right"]),
           "history leaked across the session boundary")
    second_of_a = out[out["eid"] == "a"].iloc[1]
    _check(second_of_a["prev_choice_right"] == 1.0, "lag-1 choice is wrong")
    _check(second_of_a["r_choice_prev1"] == 1.0, "reward-gated lag-1 term is wrong")
    _check(second_of_a["u_choice_prev1"] == 0.0, "unrewarded lag-1 term should be 0")
    return "add_history_features shifts within session only, gating matches the notebook"


def check_omission_handling_synthetic() -> str:
    """Omission trials must be dropped before lagging, as in the notebook."""
    frame = pd.DataFrame({
        "eid": ["a"] * 4,
        "choice": [-1, 0, 1, -1],
        "choice_right": [1.0, np.nan, 0.0, 1.0],
        "rewarded": [1.0, np.nan, 1.0, 0.0],
        "intervals_0": [0.0, 1.0, 2.0, 3.0],
    })
    out = add_history_features(frame, lags=(1,))
    _check(len(out) == 3, f"omission trial not dropped (got {len(out)} rows)")
    _check(out.iloc[1]["prev_choice_right"] == 1.0,
           "lag skipped over the omission incorrectly")
    return "omission trials dropped before lagging (3 of 4 rows retained)"


def check_fixed_rate_belief_synthetic() -> str:
    """Delta-rule belief must match a hand-computed trajectory exactly."""
    actions = np.array([1.0, 0.0, 1.0])
    pi = fixed_rate_belief(actions, np.array([0.5]), pi0=0.5)
    # pi[0]=0.5 (prior); 0.5+0.5*(1-0.5)=0.75; 0.75+0.5*(0-0.75)=0.375
    expected = np.array([0.5, 0.75, 0.375])
    _check(np.allclose(pi, expected), f"expected {expected}, got {pi}")
    return "fixed_rate_belief matches hand-computed trajectory"


def check_bayes_optimal_belief_synthetic() -> str:
    """Two structural properties any correct implementation must have:
    hazard=0.5 destroys all information every step (predictive belief is
    exactly 0.5 regardless of what is observed), and hazard~0 with consistent
    evidence should pull the belief toward the true state."""
    stim_right = np.ones(20)
    flat = bayes_optimal_belief(stim_right, np.array([0.5]))
    _check(np.allclose(flat, 0.5), f"hazard=0.5 must give a flat 0.5 belief, got range {flat.min()}-{flat.max()}")

    persistent = bayes_optimal_belief(stim_right, np.array([1e-4]))
    _check(persistent[-1] > persistent[0],
           "belief should move toward the true (right-favouring) state as evidence accumulates")
    _check(persistent[-1] > 0.7, f"near-zero hazard with 20 consistent right stimuli should end near 0.8, got {persistent[-1]}")
    _check(np.all((persistent >= 0.2) & (persistent <= 0.8)), "predictive belief left the [p_lo, p_hi] range")
    return "bayes_optimal_belief: hazard=0.5 is flat, hazard~0 tracks consistent evidence"


def check_forward_backward_synthetic() -> str:
    """Scaled forward-backward must match an unscaled brute-force computation
    (small enough sequence that the brute-force version does not underflow),
    on total log-likelihood and on the posterior at every trial."""
    trans = np.array([[0.9, 0.1], [0.2, 0.8]])
    init = np.array([0.6, 0.4])
    emis = np.array([[0.9, 0.2], [0.1, 0.8], [0.9, 0.2]])

    gamma, xi_sum, log_lik = _forward_backward(emis, trans, init)
    _check(np.allclose(gamma.sum(axis=1), 1.0), "gamma rows must sum to 1")

    # Independent unscaled reference.
    alpha_ref = [init * emis[0]]
    for t in range(1, len(emis)):
        alpha_ref.append((alpha_ref[-1] @ trans) * emis[t])
    ref_ll = float(np.log(alpha_ref[-1].sum()))
    _check(abs(log_lik - ref_ll) < 1e-9, f"log-likelihood mismatch: {log_lik} vs brute-force {ref_ll}")

    ref_gamma_last = alpha_ref[-1] / alpha_ref[-1].sum()
    _check(np.allclose(gamma[-1], ref_gamma_last),
           f"final-trial gamma mismatch: {gamma[-1]} vs brute-force {ref_gamma_last}")
    return f"forward-backward matches brute force (log-lik={log_lik:.6f})"


def check_time_block_folds_synthetic() -> str:
    """Decoding CV folds must be contiguous time blocks that partition every
    index exactly once -- a shuffled or overlapping fold would leak the
    prior's slow autocorrelation between train and test (see decode.py's
    module docstring)."""
    folds = time_block_folds(23, 5)
    all_test = np.concatenate([test for _, test in folds])
    _check(len(all_test) == 23 and set(all_test.tolist()) == set(range(23)),
           "test folds must partition all 23 indices exactly once")
    for train, test in folds:
        _check(len(set(train.tolist()) & set(test.tolist())) == 0, "train/test overlap in a fold")
        _check(bool(np.all(np.diff(test) == 1)), f"test fold not contiguous: {test}")
    return "time_block_folds: contiguous, non-overlapping, full coverage over 23 indices / 5 folds"


def check_ridge_r2_skips_degenerate_fold() -> str:
    """A test fold whose target is constant must be skipped, not scored --
    R^2 explodes to an arbitrarily large negative number there (SS_tot -> 0),
    which found a real -1.29e8 fold in actual decode.py output before this
    guard existed. Built with 100 trials so time_block_folds(100, 5) gives
    exactly 5 contiguous 20-trial blocks, and the third one made exactly
    constant."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 5))
    # Square-wave, not a smooth ramp: a real pi_hat oscillates between its two
    # extremes several times within a session, so each contiguous 20-trial
    # block should retain plenty of variance -- a monotonic ramp chunked into
    # blocks gives each block a narrow, low-variance slice and was the first
    # version of this test, which (correctly, but not for the intended
    # reason) also tripped the guard on the "good" blocks.
    y = np.where((np.arange(n) // 5) % 2 == 0, 0.2, 0.8) + rng.normal(scale=0.01, size=n)
    y[40:60] = 0.5  # exactly one contiguous outer test block forced constant

    folds = _ridge_r2_nested_cv(X, y, alphas=(1.0, 10.0), n_outer=5, n_inner=3)
    _check(len(folds) < 5, "the degenerate block's fold should have been skipped, not scored")
    r2_values = [f["r2"] for f in folds]
    _check(all(np.isfinite(r2_values)), f"non-finite R^2 slipped through: {r2_values}")
    _check(all(r2 > -100 for r2 in r2_values),
           f"an exploded R^2 slipped through despite the guard: {r2_values}")
    return f"_ridge_r2_nested_cv skips the constant-target fold ({len(folds)}/5 folds scored)"


def check_window_mean_synthetic() -> str:
    """Windowed averaging must respect the window and report NaN outside data."""
    from .covariates import _window_mean

    ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    vals = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    # Window [0.6, 1.0] around an event at 1.0 covers 0.75 and 1.0 -> mean 35.
    means = _window_mean(ts, vals, np.array([1.0]), (-0.4, 0.0))
    _check(means[0] == 35.0, f"expected mean 35.0, got {means[0]}")

    # An event whose window predates every sample has no data, not a zero.
    before = _window_mean(ts, vals, np.array([-5.0]), (-0.4, 0.0))
    _check(np.isnan(before[0]), "window before all samples must be NaN, not 0")
    return "_window_mean averages inside the window and returns NaN when unsampled"


# --- online checks --------------------------------------------------------

def check_trial_counts_against_one(one: Any, eid: str) -> str:
    """Extracted trial count must reconcile with the raw ONE trials table."""
    raw = load_trials(one, eid)
    extracted = extract_session_behavior(one, eid, "_iblrig_tasks_ephysChoiceWorld6.4.2")

    n_omissions = int((raw["choice"] == 0).sum())
    _check(len(extracted) == len(raw) - n_omissions,
           f"trial count mismatch: {len(extracted)} != {len(raw)} - {n_omissions}")

    # Independent re-derivation of the sign convention on this session.
    correct = raw[raw["feedbackType"] == 1]
    right_stim = correct.dropna(subset=["contrastRight"])
    _check(set(right_stim["choice"].unique()) <= {-1},
           "choice sign convention violated: correct right-stimulus trials must be -1")
    right_mapped = extracted[extracted["signed_contrast"] > 0]
    _check(right_mapped["choice_right"].isin([0.0, 1.0]).all(),
           "choice_right contains values outside {0,1}")
    return (f"{eid[:8]}: {len(raw)} trials in ONE, {n_omissions} omissions, "
            f"{len(extracted)} extracted -- reconciles; sign convention holds")


def check_cluster_counts_against_one(one: Any, pid: str) -> str:
    """Cluster table must match the raw clusters/channels objects from ONE."""
    from brainbox.io.one import SpikeSortingLoader

    table = load_cluster_table(one, pid)
    loader = SpikeSortingLoader(pid=pid, one=one)
    raw_clusters = loader.load_spike_sorting_object("clusters")
    raw_channels = loader.load_channels()

    n_raw = len(np.asarray(raw_clusters["channels"]))
    _check(len(table) == n_raw, f"cluster count mismatch: {len(table)} != {n_raw}")

    raw_label = np.asarray(raw_clusters["metrics"]["label"], dtype=float)
    n_pass_raw = int((raw_label >= config.QC_LABEL_PASS).sum())
    _check(int(table["qc_pass"].sum()) == n_pass_raw,
           f"QC-pass count mismatch: {int(table['qc_pass'].sum())} != {n_pass_raw}")

    # Region assignment: cluster acronym must equal its peak channel's acronym.
    raw_acronyms = np.asarray(raw_channels["acronym"], dtype=object)
    channel_idx = np.asarray(raw_clusters["channels"]).astype(int)
    expected = raw_acronyms[np.clip(channel_idx, 0, raw_acronyms.size - 1)]
    _check(bool((table["acronym"].to_numpy() == expected).all()),
           "per-cluster region acronym does not match its peak channel")
    return (f"{pid[:8]}: {n_raw} clusters, {n_pass_raw} QC-pass -- matches ONE; "
            "region labels track peak channel")


def check_region_hierarchy(one: Any, pid: str) -> str:
    """Hierarchy matching must not be defeated by cortical layer suffixes."""
    table = load_cluster_table(one, pid)
    visp_ids = region_atlas_ids("VISp")
    _check(len(visp_ids) > 1, "VISp descendant set should include layer subdivisions")
    layered = {a for a in table["acronym"].unique() if isinstance(a, str) and any(
        ch.isdigit() for ch in a)}
    matched = table[table["atlas_id"].isin(region_atlas_ids("VISp"))]
    return (f"VISp expands to {len(visp_ids)} CCF ids; this insertion has "
            f"{len(layered)} layered acronyms, {len(matched)} clusters matched to VISp")


def check_wheel_velocity_against_one(one: Any, eid: str) -> str:
    """Wheel velocity must be a real velocity, not the differentiated time axis.

    `interpolate_position` returns (position, time); unpacking it backwards makes
    velocity_filtered differentiate t against itself, giving a constant 1.0 rad/s
    on every trial. That is silent -- the column is fully populated and looks
    fine -- so it is asserted here rather than left to inspection.
    """
    from .covariates import _wheel_velocity, extract_covariates

    trials = load_trials(one, eid)
    events = trials["stimOn_times"].dropna().to_numpy(dtype=float)
    timestamps, velocity = _wheel_velocity(one, eid)

    _check(np.all(np.diff(timestamps) > 0),
           "wheel timestamps are not monotonically increasing (position/time swapped?)")
    _check(float(np.std(velocity)) > 1e-6,
           f"wheel velocity is constant (std={np.std(velocity):.2e}) -- axes swapped")
    _check(not np.allclose(velocity, 1.0),
           "wheel velocity is identically 1.0 -- the time axis was differentiated")

    covariates, status = extract_covariates(one, eid, events)
    _check(status["wheel"] == "ok", f"wheel covariate failed: {status['wheel']}")
    coverage = float(covariates["wheel_speed_mean"].notna().mean())
    _check(coverage > 0.9,
           f"wheel covers only {coverage:.1%} of trials -- timestamps likely wrong")
    return (f"{eid[:8]}: wheel velocity std={np.std(velocity):.3f} rad/s, "
            f"covers {coverage:.1%} of trials")


def check_spike_counts_against_one(one: Any, pid: str, eid: str, n_trials: int = 20) -> str:
    """Recompute a few trials' counts with a naive mask and compare.

    The fast path uses searchsorted; the reference here is the obvious (slow)
    boolean-mask computation, so the two disagreeing means the optimisation is
    wrong.
    """
    trials = load_trials(one, eid)
    events = trials["stimOn_times"].dropna().to_numpy(dtype=float)[:n_trials]
    table = load_cluster_table(one, pid)
    good = table[table["qc_pass"]].head(5)
    if good.empty:
        return f"{pid[:8]}: no QC-passing clusters, spike-count check skipped"

    times, clusters = load_spike_trains(one, pid)
    cluster_ids = good["cluster_id"].to_numpy(dtype=int)
    fast = prestim_spike_counts(times, clusters, cluster_ids, events, config.PRESTIM_WINDOW_S)

    start, end = config.PRESTIM_WINDOW_S
    slow = np.zeros_like(fast)
    for i, event in enumerate(events):
        in_window = (times >= event + start) & (times <= event + end)
        for j, cid in enumerate(cluster_ids):
            slow[i, j] = int(np.sum(in_window & (clusters == cid)))

    _check(np.array_equal(fast, slow),
           f"fast and reference spike counts disagree (max diff {np.abs(fast - slow).max()})")
    return (f"{pid[:8]}: searchsorted counts == brute-force counts over "
            f"{len(events)} trials x {len(cluster_ids)} clusters (total "
            f"{int(fast.sum())} spikes)")


def run(offline_only: bool = False, eid: str = DEFAULT_EID, pid: str = DEFAULT_PID) -> int:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("prestim counts (synthetic)", check_prestim_counts_synthetic),
        ("history features (synthetic)", check_history_features_synthetic),
        ("omission handling (synthetic)", check_omission_handling_synthetic),
        ("window mean (synthetic)", check_window_mean_synthetic),
        ("fixed-rate belief (synthetic)", check_fixed_rate_belief_synthetic),
        ("bayes-optimal belief (synthetic)", check_bayes_optimal_belief_synthetic),
        ("forward-backward (synthetic)", check_forward_backward_synthetic),
        ("time-block folds (synthetic)", check_time_block_folds_synthetic),
        ("ridge R^2 degenerate-fold guard (synthetic)", check_ridge_r2_skips_degenerate_fold),
    ]

    if not offline_only:
        from .one_client import get_one
        one = get_one()
        checks += [
            ("trial counts vs ONE", lambda: check_trial_counts_against_one(one, eid)),
            ("cluster counts vs ONE", lambda: check_cluster_counts_against_one(one, pid)),
            ("region hierarchy", lambda: check_region_hierarchy(one, pid)),
            ("wheel velocity vs ONE", lambda: check_wheel_velocity_against_one(one, eid)),
            ("spike counts vs brute force", lambda: check_spike_counts_against_one(one, pid, eid)),
        ]

    failures = 0
    for name, check in checks:
        try:
            detail = check()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}\n        {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"PASS  {name}\n        {detail}", flush=True)

    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="skip checks needing ONE")
    parser.add_argument("--eid", default=DEFAULT_EID)
    parser.add_argument("--pid", default=DEFAULT_PID)
    args = parser.parse_args(argv)
    return run(offline_only=args.offline, eid=args.eid, pid=args.pid)


if __name__ == "__main__":
    sys.exit(main())
