"""Pseudo-session generation for the Stage 2 null test (proposal Q8, step 7).

A decoded R^2 for the block prior cannot be read against zero. Two things
inflate it that have nothing to do with prior coding:

  * the target is slowly autocorrelated, so interleaved CV folds put trials
    that are neighbours in time on both sides of the split;
  * firing rates drift across a session, and blocks are contiguous in time,
    so "when in the session am I" is partly predictive of "which block am I
    in". Measured on this project's own data, trial index decodes from
    pre-stimulus counts at median R^2 = 0.64 -- drift is large, not marginal.

The pseudo-session null absorbs both. A pseudo-session is a fresh block
sequence drawn from the *same generative process the rig used*, paired with
the *real, untouched* neural data. It therefore carries the identical
autocorrelation and the identical contiguity-vs-drift relationship as the
real target, but by construction cannot be causally related to the activity.
The p-value is the fraction of pseudo-sessions scoring at least as high as
the real one.

The process parameters below were measured from this project's own 84
extracted sessions rather than taken from the task spec, so the null matches
the data actually being tested:

  first block   : 90 trials at pL = 0.5 in 84/84 sessions.
  later blocks  : strictly alternate 0.2 / 0.8; the first of them is 0.8 in
                  48/84 sessions and 0.2 in 36 -- consistent with a fair coin
                  (two-sided binomial p = 0.22), so it is drawn 50/50 here.
  block lengths : truncated exponential on [20, 100]. 907 interior blocks
                  gave mean 50.0, median 44; scale 50 simulates to mean 49.7,
                  the closest of the scales tried (55 -> 50.6, 60 -> 51.3).
                  A handful of real blocks fall just outside [20, 100]
                  (observed min 18, max 99); the bounds are kept at the task's
                  nominal values rather than widened to chase two outliers.
"""
from __future__ import annotations

import numpy as np

FIRST_BLOCK_LEN = 90
FIRST_BLOCK_PL = 0.5
BLOCK_PL_VALUES = (0.2, 0.8)
BLOCK_LEN_SCALE = 50.0
BLOCK_LEN_MIN = 20
BLOCK_LEN_MAX = 100


def draw_block_length(rng: np.random.Generator) -> int:
    """One block length from an exponential truncated to [20, 100].

    Rejection sampling, not clipping: clipping would pile probability mass
    onto the two bounds themselves, which the real length histogram does not
    show. The loop is bounded because the acceptance region carries ~55% of
    an exponential(50), so it terminates in a couple of draws in practice.
    """
    while True:
        length = int(rng.exponential(BLOCK_LEN_SCALE))
        if BLOCK_LEN_MIN <= length <= BLOCK_LEN_MAX:
            return length


def generate_block_sequence(n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """A pL sequence of exactly `n_trials`, drawn the way the rig draws it.

    The final block is truncated by the end of the session, exactly as a real
    session's last block is -- so the null inherits the real data's slightly
    short trailing block rather than an idealised complete one.
    """
    pl = np.full(n_trials, FIRST_BLOCK_PL, dtype=float)
    if n_trials <= FIRST_BLOCK_LEN:
        return pl

    idx = FIRST_BLOCK_LEN
    value = BLOCK_PL_VALUES[rng.integers(2)]
    while idx < n_trials:
        length = draw_block_length(rng)
        pl[idx:idx + length] = value
        idx += length
        value = BLOCK_PL_VALUES[0] if value == BLOCK_PL_VALUES[1] else BLOCK_PL_VALUES[1]
    return pl


def generate_pseudo_targets(n_trials: int, n_pseudo: int, rng: np.random.Generator,
                            min_std: float = 1e-6) -> np.ndarray:
    """`n_pseudo` x `n_trials` matrix of independent pseudo-session pL sequences.

    A draw with no variance (possible only for a session shorter than the
    90-trial opening block, which the callers already exclude) is rejected
    rather than returned, because it cannot be scored by R^2.
    """
    out = np.empty((n_pseudo, n_trials), dtype=float)
    filled = 0
    while filled < n_pseudo:
        candidate = generate_block_sequence(n_trials, rng)
        if candidate.std() < min_std:
            continue
        out[filled] = candidate
        filled += 1
    return out


def generate_stimulus_sides(block_sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stimulus sides (1 = right) drawn from a block sequence's own probabilities.

    `probabilityLeft` is the probability the stimulus appears on the LEFT, so
    P(right) = 1 - pL. This is the second half of the counterfactual the IBL
    prior analyses use: a null for a *belief* target cannot be built by
    shuffling the belief, because the belief is a deterministic function of the
    stimulus sequence. The stimulus sequence is what has to be regenerated, and
    the counterfactual belief is then computed from it exactly as the real one
    was from the real sequence.
    """
    return (rng.random(len(block_sequence)) >= block_sequence).astype(float)
