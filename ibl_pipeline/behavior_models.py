"""Stage 1: behavioural model fitting (proposal Q8, Models 1 and 2).

Two independent pieces, both operating on the trial-level table `behavior.py`
already produces (so nothing here re-derives signed_contrast, choice_right,
prior_bias_right, or the history-lag columns):

- Model 1 (`history_glm_*`): a single pooled logistic regression predicting
  choice from contrast, block prior, and reward-gated choice history at lags
  1..5, L2-regularised with the strength chosen by nested cross-validation
  (folds split at the *session* level so a fold never sees half of a session).

- Model 2 (`fit_belief_model` / `compare_belief_models`): four competing
  single-trial-update models of how a "prior" belief evolves trial to trial,
  fit per session by maximum likelihood and compared by held-out
  log-likelihood on a temporal (not shuffled) train/test split within each
  session -- shuffling would leak future trials backward through the belief
  recursion, which is exactly the kind of silent bug `sanity_check.py` exists
  to catch, so this module is deliberately tested against synthetic sequences
  the same way.

- `fit_glm_hmm`: a 3-state Bernoulli GLM-HMM (engaged / left-biased /
  right-biased), used to flag which trials belong to the "engaged" state
  before Model 2's comparison is run on the restricted trial set (proposal
  step 5).

One deliberate deviation from the proposal's LaTeX, stated once here rather
than buried in a comment on every function: the proposal writes the belief
update as pi_{t+1} = pi_t + alpha*(a_t - pi_t) with a_t in {-1,+1}, then reuses
pi_t as logit(pi_t) in the choice policy. That recursion does not keep pi_t in
(0,1) for a_t in {-1,+1} (a convex combination only stays in the interval its
inputs are drawn from), so logit(pi_t) is undefined in general under a literal
reading. Here a_t is `choice_right` in {0,1} instead, so pi_t is exactly "the
believed probability the next choice is rightward", stays in [0,1] by
construction, and logit(pi_t) is always defined. The qualitative model
(an exponentially-weighted trace of past actions) is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit

from . import config

_EPS = 1e-6


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _bernoulli_ll_per_trial(y: np.ndarray, p: np.ndarray) -> float:
    """Mean Bernoulli log-likelihood per trial; the metric comparable across
    sessions/folds of different sizes (a raw sum is not)."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# Model 1: history-dependent logistic regression
# ---------------------------------------------------------------------------

HISTORY_GLM_REGRESSORS: tuple[str, ...] = ("signed_contrast", "prior_bias_right") + tuple(
    col for lag in config.HISTORY_LAGS for col in (f"r_choice_prev{lag}", f"u_choice_prev{lag}")
)


def history_glm_design(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, y, session_groups), NaN-dropped, restricted to block-structured trials.

    Block-restriction matters here specifically because `prior_bias_right` is
    meaningless outside biased/ephys sessions (notebook practical issue #4;
    `config.INCLUDED_PROTOCOL_STAGES` is the same constant `insertions.py` uses
    for the identical reason).
    """
    block = trials[trials["stage"].isin(config.INCLUDED_PROTOCOL_STAGES)]
    usable = block.dropna(subset=list(HISTORY_GLM_REGRESSORS) + ["choice_right"])
    return (usable[list(HISTORY_GLM_REGRESSORS)].reset_index(drop=True),
            usable["choice_right"].reset_index(drop=True),
            usable["eid"].reset_index(drop=True))


def _group_kfold_indices(groups: pd.Series, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by session, shuffled at the group level.

    Implemented directly (not sklearn.GroupKFold) so the split is randomised --
    sklearn's GroupKFold is deterministic in first-seen order, which for our
    eids means an accidental split by acquisition date rather than a random
    sample.
    """
    # Plain numpy object array: shuffling a pandas ExtensionArray (e.g. Arrow-backed
    # strings, which is what eids sometimes come back as) is only "not guaranteed to
    # behave correctly" per numpy's own warning -- silently risking duplicate eids
    # across folds, which would leak a session's trials into both train and test.
    unique_groups = np.array(groups.unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    folds = np.array_split(unique_groups, n_splits)
    idx = np.arange(len(groups))
    splits = []
    for k in range(n_splits):
        test_groups = set(folds[k])
        test_mask = groups.isin(test_groups).to_numpy()
        splits.append((idx[~test_mask], idx[test_mask]))
    return splits


def fit_history_glm_nested_cv(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
    Cs: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    n_outer: int = 5, n_inner: int = 4, seed: int = 0,
) -> pd.DataFrame:
    """Nested session-grouped CV: outer loop reports generalisation, inner
    loop only ever picks the L2 strength (it never touches the outer test
    fold), so the reported log-likelihood is not optimistic from tuning C on
    the same data it is scored on.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xv, yv = X.to_numpy(), y.to_numpy()
    rows = []
    for outer_i, (train_idx, test_idx) in enumerate(_group_kfold_indices(groups, n_outer, seed)):
        g_train = groups.iloc[train_idx]
        inner_splits = _group_kfold_indices(g_train.reset_index(drop=True), n_inner, seed + 1)

        best_C, best_ll = Cs[0], -np.inf
        for C in Cs:
            fold_lls = []
            for tr2, va2 in inner_splits:
                tr_abs, va_abs = train_idx[tr2], train_idx[va2]
                scaler = StandardScaler().fit(Xv[tr_abs])
                clf = LogisticRegression(C=C, max_iter=2000).fit(
                    scaler.transform(Xv[tr_abs]), yv[tr_abs])
                p = clf.predict_proba(scaler.transform(Xv[va_abs]))[:, 1]
                fold_lls.append(_bernoulli_ll_per_trial(yv[va_abs], p))
            mean_ll = float(np.mean(fold_lls))
            if mean_ll > best_ll:
                best_ll, best_C = mean_ll, C

        scaler = StandardScaler().fit(Xv[train_idx])
        clf = LogisticRegression(C=best_C, max_iter=2000).fit(
            scaler.transform(Xv[train_idx]), yv[train_idx])
        p_test = clf.predict_proba(scaler.transform(Xv[test_idx]))[:, 1]
        rows.append({
            "outer_fold": outer_i, "best_C": best_C,
            "test_ll_per_trial": _bernoulli_ll_per_trial(yv[test_idx], p_test),
            "test_accuracy": float(np.mean((p_test > 0.5) == yv[test_idx])),
            "n_test_trials": int(test_idx.size),
            "n_test_sessions": int(groups.iloc[test_idx].nunique()),
        })
    return pd.DataFrame(rows)


def fit_history_glm_interpretable(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Any:
    """Unregularised fit on the full pilot set with session-clustered robust
    SEs, for coefficients a human reads -- separate from the nested-CV model
    above, whose L2 shrinkage makes its raw coefficients not directly
    interpretable as effect sizes.
    """
    import statsmodels.api as sm

    Xc = sm.add_constant(X)
    return sm.Logit(y, Xc).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})


# ---------------------------------------------------------------------------
# Model 2: trial-updating belief models
# ---------------------------------------------------------------------------

def fixed_rate_belief(action_right: np.ndarray, params: np.ndarray, pi0: float = 0.5) -> np.ndarray:
    """pi[t] = belief entering trial t, i.e. *before* action_right[t]."""
    (alpha,) = params
    n = action_right.size
    pi = np.empty(n)
    cur = pi0
    for t in range(n):
        pi[t] = cur
        cur = cur + alpha * (action_right[t] - cur)
    return pi


def two_rate_belief(action_right: np.ndarray, rewarded: np.ndarray, params: np.ndarray, pi0: float = 0.5) -> np.ndarray:
    alpha_pos, alpha_neg = params
    n = action_right.size
    pi = np.empty(n)
    cur = pi0
    for t in range(n):
        pi[t] = cur
        a = alpha_pos if rewarded[t] else alpha_neg
        cur = cur + a * (action_right[t] - cur)
    return pi


def surprise_scaled_belief(action_right: np.ndarray, params: np.ndarray, pi0: float = 0.5) -> np.ndarray:
    alpha0, alpha1 = params
    n = action_right.size
    pi = np.empty(n)
    cur = pi0
    for t in range(n):
        pi[t] = cur
        delta = action_right[t] - cur
        alpha = expit(alpha0 + alpha1 * abs(delta))
        cur = cur + alpha * delta
    return pi


def bayes_optimal_belief(stim_right: np.ndarray, params: np.ndarray,
                          p_lo: float = 0.2, p_hi: float = 0.8) -> np.ndarray:
    """Exact forward filter over a 2-state hazard process on the block prior.

    Unlike the three delta-rule models above, this one conditions on the
    *actual stimulus side* each trial, not the animal's action -- it is the
    normative belief a Bayesian observer would hold about the experimenter's
    hidden block variable, which the delta-rule models never have access to.
    That is a real, reportable distinction between the models, not a detail:
    it is the difference between "tracks its own action history" and "tracks
    the true generative statistics", both of which the proposal lists as
    live hypotheses.

    Returns the *predictive* P(right stimulus) entering each trial, before
    that trial's own stimulus is observed -- the quantity comparable to the
    other three models' pi[t].
    """
    (hazard,) = params
    hazard = float(np.clip(hazard, _EPS, 0.5))
    n = stim_right.size
    pred = np.empty(n)
    post_hi = 0.5  # belief P(state == hi) after the previous trial's stimulus
    for t in range(n):
        pred_hi = post_hi * (1 - hazard) + (1 - post_hi) * hazard
        pred[t] = pred_hi * p_hi + (1 - pred_hi) * p_lo
        lik_hi = p_hi if stim_right[t] else (1 - p_hi)
        lik_lo = p_lo if stim_right[t] else (1 - p_lo)
        num = lik_hi * pred_hi
        den = num + lik_lo * (1 - pred_hi)
        post_hi = num / den if den > _EPS else pred_hi
    return pred


@dataclass(frozen=True)
class BeliefModel:
    name: str
    n_belief_params: int
    belief_bounds: tuple[tuple[float, float], ...]
    belief_fn: Callable[..., np.ndarray]
    uses: str  # "action" or "stimulus" -- documents what drives the belief update


BELIEF_MODELS: dict[str, BeliefModel] = {
    "fixed_rate": BeliefModel("fixed_rate", 1, ((_EPS, 1 - _EPS),),
                               lambda a, r, s, p: fixed_rate_belief(a, p), "action"),
    "two_rate": BeliefModel("two_rate", 2, ((_EPS, 1 - _EPS), (_EPS, 1 - _EPS)),
                             lambda a, r, s, p: two_rate_belief(a, r, p), "action"),
    "surprise_scaled": BeliefModel("surprise_scaled", 2, ((-5.0, 5.0), (-10.0, 10.0)),
                                    lambda a, r, s, p: surprise_scaled_belief(a, p), "action"),
    "bayes_optimal": BeliefModel("bayes_optimal", 1, ((_EPS, 0.5),),
                                  lambda a, r, s, p: bayes_optimal_belief(s, p), "stimulus"),
}

# Shared choice policy: P(right) = lapse/2 + (1-lapse) * sigmoid(gamma*contrast
# + eta*logit(belief)). 3 extra params on top of each belief model's own.
POLICY_BOUNDS: tuple[tuple[float, float], ...] = ((-50.0, 50.0), (-20.0, 20.0), (0.0, 0.3))


def _policy_prob(contrast: np.ndarray, belief: np.ndarray, gamma: float, eta: float, lapse: float) -> np.ndarray:
    core = expit(gamma * contrast + eta * _logit(belief))
    return lapse / 2 + (1 - lapse) * core


def _session_arrays(trials: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pull the arrays every belief model needs, in temporal order, omission-free.

    `trials` here is already the output of `add_history_features`, which drops
    omissions -- so this is a pure re-sort, not a second filtering pass.
    """
    sort_key = "intervals_0" if "intervals_0" in trials.columns else "stimOn_times"
    t = trials.sort_values(sort_key)
    return {
        "action_right": t["choice_right"].to_numpy(dtype=float),
        "rewarded": t["rewarded"].to_numpy(dtype=float),
        "stim_right": t["contrastRight"].notna().to_numpy(dtype=float),
        "contrast": t["signed_contrast"].to_numpy(dtype=float),
        "choice_right": t["choice_right"].to_numpy(dtype=float),
    }


def _nll(params: np.ndarray, model: BeliefModel, arrs: dict[str, np.ndarray],
          fit_idx: np.ndarray, l2: float = 0.0) -> float:
    belief_params = params[: model.n_belief_params]
    gamma, eta, lapse = params[model.n_belief_params:]
    belief = model.belief_fn(arrs["action_right"], arrs["rewarded"], arrs["stim_right"], belief_params)
    p = _policy_prob(arrs["contrast"][fit_idx], belief[fit_idx], gamma, eta, lapse)
    y = arrs["choice_right"][fit_idx]
    p = np.clip(p, _EPS, 1 - _EPS)
    data_term = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    # l2=0 by default -- see fit_belief_model's docstring for why. Available
    # as an option (weak Gaussian prior on gamma/eta) for whoever wants to
    # explore it, but it is not a default-on fix.
    reg_term = 0.5 * l2 * (gamma ** 2 + eta ** 2)
    return float(data_term + reg_term)


def fit_belief_model(model_name: str, trials: pd.DataFrame, test_frac: float = 0.2,
                      n_restarts: int = 4, seed: int = 0, l2: float = 0.0) -> dict[str, Any] | None:
    """Fit one belief model on one session's trials; temporal train/test split.

    The split is temporal, not shuffled: the belief recursion is stateful, so
    a shuffled split would let test-fold trials leak into the belief the model
    carries into training trials computed after them. The belief is always
    computed over the *whole* session (so training itself sees a continuous,
    uninterrupted trace) and only the log-likelihood is restricted to each
    fold's trial indices.

    Returns None (rather than raising) for a session too short to fit, so a
    caller aggregating across many sessions can filter cleanly instead of
    wrapping every call in try/except.

    `l2` defaults to 0 -- unregularised -- on purpose, after actually testing
    the alternative. About half of the 79-session pilot's bayes_optimal fits
    land on hazard=0.5, the upper bound; there, logit(0.5)=0 makes eta have
    zero effect on the likelihood, so unconstrained MLE parks eta at *its
    own* boundary arbitrarily. Adding an L2 prior on gamma/eta does pull
    hazard off the 0.5 boundary, but on the one session this was checked
    against directly, held-out test log-likelihood got monotonically *worse*
    as l2 increased (-0.168 at l2=0, -0.420 at l2=5) -- the boundary fit is
    not an optimisation failure, it is the best-predicting description of
    that session's engaged-trial choices. Regularising it away would mean
    asserting a prior-tracking signal the data does not support. The correct
    response to a session like this is to exclude it from anything that needs
    a non-degenerate belief trajectory (see decode.py's MIN_PI_HAT_STD guard),
    not to manufacture one.
    """
    model = BELIEF_MODELS[model_name]
    arrs = _session_arrays(trials)
    n = arrs["contrast"].size
    n_test = int(round(n * test_frac))
    if n < 40 or n_test < 10 or (n - n_test) < 20:
        return None
    train_idx = np.arange(0, n - n_test)
    test_idx = np.arange(n - n_test, n)

    bounds = model.belief_bounds + POLICY_BOUNDS
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        try:
            # l2 shapes the *optimisation* (breaks the flat/degenerate
            # direction described in _nll's docstring); it must not leak into
            # the reported likelihoods below, which need to be the plain data
            # term for train_ll/test_ll/BIC to mean what they say and stay
            # comparable across models with different l2 sensitivity.
            res = optimize.minimize(_nll, x0, args=(model, arrs, train_idx, l2),
                                     method="L-BFGS-B", bounds=bounds)
        except Exception:  # noqa: BLE001 - a bad random start must not kill the session
            continue
        if res.success:
            unpenalised = _nll(res.x, model, arrs, train_idx, l2=0.0)
            if best is None or unpenalised < best[1]:
                best = (res, unpenalised)
    if best is None:
        return None
    res, train_nll = best

    k = len(bounds)
    test_nll = _nll(res.x, model, arrs, test_idx, l2=0.0)
    return {
        "model": model_name, "uses": model.uses,
        "n_params": k, "params": res.x.tolist(),
        "n_train": int(train_idx.size), "n_test": int(test_idx.size),
        "train_ll_per_trial": -train_nll / train_idx.size,
        "test_ll_per_trial": -test_nll / test_idx.size,
        "bic": k * np.log(train_idx.size) + 2 * train_nll,
    }


def compare_belief_models(sessions: dict[str, pd.DataFrame], test_frac: float = 0.2,
                           n_restarts: int = 4, seed: int = 0, l2: float = 0.0) -> pd.DataFrame:
    """Fit every model in `BELIEF_MODELS` on every session; one row per
    (session, model) that converged. Sessions too short simply contribute no
    rows for that fit, rather than NaNs that would silently pull down a mean.
    """
    rows = []
    for eid, trials in sessions.items():
        for name in BELIEF_MODELS:
            fit = fit_belief_model(name, trials, test_frac=test_frac,
                                    n_restarts=n_restarts, seed=seed, l2=l2)
            if fit is not None:
                fit["eid"] = eid
                rows.append(fit)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GLM-HMM: 3-state engaged / left-biased / right-biased gating (proposal step 5)
# ---------------------------------------------------------------------------

def _state_logistic_prob(contrast: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """weights = [bias, contrast_slope] per state."""
    return expit(weights[0] + weights[1] * contrast)


def _fit_glm_hmm_once(contrast: np.ndarray, y: np.ndarray, seq_bounds: list[tuple[int, int]],
                       n_states: int, n_iter: int, tol: float, l2: float, sticky_kappa: float,
                       rng: np.random.Generator) -> dict[str, Any]:
    weights = rng.normal(scale=0.5, size=(n_states, 2))
    trans = np.full((n_states, n_states), 1.0 / n_states)
    init = np.full(n_states, 1.0 / n_states)

    prev_ll = -np.inf
    gammas = np.empty((len(y), n_states))
    for iteration in range(n_iter):
        emis = np.stack([
            np.where(y == 1, _state_logistic_prob(contrast, weights[k]),
                      1 - _state_logistic_prob(contrast, weights[k]))
            for k in range(n_states)
        ], axis=1)  # (n_trials, n_states)
        emis = np.clip(emis, _EPS, 1.0)

        xi_sum = np.zeros((n_states, n_states))
        total_ll = 0.0
        for start, end in seq_bounds:
            g, xi, ll = _forward_backward(emis[start:end], trans, init)
            gammas[start:end] = g
            xi_sum += xi
            total_ll += ll

        # M-step: transition matrix (MAP under a Dirichlet prior with extra
        # pseudocounts `sticky_kappa` on the diagonal -- see fit_glm_hmm's
        # docstring for why an unbiased MLE here fits noise-level switching
        # instead of the slowly-persisting behavioural states these are meant
        # to capture) and initial distribution from posteriors ...
        sticky = sticky_kappa * np.eye(n_states)
        trans = (xi_sum + sticky) / np.clip(
            xi_sum.sum(axis=1, keepdims=True) + sticky_kappa, _EPS, None)
        init = np.mean([gammas[s] for s, _ in seq_bounds], axis=0)

        # ... and each state's logistic weights, by weighted MLE.
        for k in range(n_states):
            weights[k] = _weighted_logistic_fit(contrast, y, gammas[:, k], weights[k], l2=l2)

        if abs(total_ll - prev_ll) < tol * abs(prev_ll or 1.0):
            prev_ll = total_ll
            break
        prev_ll = total_ll

    return {"weights": weights, "trans": trans, "init": init, "gammas": gammas,
            "log_likelihood": prev_ll, "n_iter": iteration + 1, "converged": iteration + 1 < n_iter}


def fit_glm_hmm(trials: pd.DataFrame, n_states: int = 3, n_iter: int = 50,
                 tol: float = 1e-4, l2: float = 4.0, sticky_kappa: float = 200.0,
                 n_restarts: int = 6, seed: int = 0) -> dict[str, Any]:
    """3-state Bernoulli GLM-HMM via EM, pooled across all sessions passed in.

    Each state k has its own logistic weights [bias_k, contrast_slope_k] on
    choice_right; states are expected to separate into one with a steep
    contrast slope and near-zero bias ("engaged") and two with a shallow slope
    and a large bias of either sign ("biased left/right" or disengaged), which
    is the qualitative signature reported for this task elsewhere -- but the
    labelling is done in `label_engaged_state` from the fitted weights, not
    assumed here.

    Two failure modes this guards against, both found by actually running the
    fit on real data rather than assumed up front:

    - EM is notoriously multimodal here (random weight init + a handful of
      trials driving one state's posterior can lock onto a bad optimum), so
      this restarts `n_restarts` times from independent random initialisations
      and keeps the highest final log-likelihood -- the first version of this
      function used a single fixed-seed run and silently returned whichever
      local optimum that seed happened to find.
    - The per-state weighted logistic fit is otherwise unregularised, and a
      state whose effective (posterior-weighted) sample is small and close to
      linearly separable in contrast will drive its slope to a very large,
      poorly-identified value (quasi-separation -- the same failure mode noted
      for the full-contrast pooled GLM in the exploratory notebook, but here
      unmitigated it also starves that state's posterior mass, since an
      extreme, near-deterministic emission model fits almost no trials well).
      `l2` is a weak Gaussian prior pulling weights toward 0, breaking the
      blow-up without needing to hand-pick which state it happens to hit.

    A third, structural fix: `sticky_kappa` adds Dirichlet pseudocounts to the
    transition matrix's diagonal (a standard "sticky HMM" prior). Without it,
    an unbiased M-step happily fits a transition matrix that switches state
    almost every trial -- which explains a Bernoulli sequence just as well in
    likelihood terms, but is not what "the animal is in an engaged epoch for
    dozens of trials" means, and was confirmed empirically here: with
    sticky_kappa=0 the fitted diagonal was 0.13-0.63 and no state's posterior
    ever exceeded ~0.7.

    Sequence boundaries make forward-backward operate per session, not across
    the pooled trials: history within a session is a real Markov chain, but
    trial 1 of session B has no dependence on the last trial of session A.
    """
    contrast = trials["signed_contrast"].to_numpy(dtype=float)
    y = trials["choice_right"].to_numpy(dtype=float)
    eids = trials["eid"].to_numpy()
    seq_bounds = _session_boundaries(eids)

    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        fit = _fit_glm_hmm_once(contrast, y, seq_bounds, n_states, n_iter, tol, l2,
                                 sticky_kappa, rng)
        if best is None or fit["log_likelihood"] > best["log_likelihood"]:
            best = fit
    best["n_restarts"] = n_restarts
    return best


def _session_boundaries(eids: np.ndarray) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for i in range(1, len(eids) + 1):
        if i == len(eids) or eids[i] != eids[start]:
            bounds.append((start, i))
            start = i
    return bounds


def _forward_backward(emis: np.ndarray, trans: np.ndarray, init: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """Scaled forward-backward for one sequence; returns (gamma, xi_sum, log-lik)."""
    n, k = emis.shape
    alpha = np.empty((n, k))
    scale = np.empty(n)
    alpha[0] = init * emis[0]
    scale[0] = alpha[0].sum() or _EPS
    alpha[0] /= scale[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ trans) * emis[t]
        scale[t] = alpha[t].sum() or _EPS
        alpha[t] /= scale[t]

    beta = np.empty((n, k))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (trans @ (emis[t + 1] * beta[t + 1])) / scale[t + 1]

    gamma = alpha * beta
    gamma /= np.clip(gamma.sum(axis=1, keepdims=True), _EPS, None)

    xi_sum = np.zeros((k, k))
    for t in range(n - 1):
        num = (alpha[t][:, None] * trans) * (emis[t + 1] * beta[t + 1])[None, :]
        xi_sum += num / (num.sum() or _EPS)

    log_lik = float(np.sum(np.log(scale)))
    return gamma, xi_sum, log_lik


def _weighted_logistic_fit(contrast: np.ndarray, y: np.ndarray, weight: np.ndarray,
                            x0: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """Weighted 1-covariate logistic regression via direct NLL optimisation.

    `l2` adds 0.5*l2*||w||^2 to the objective: a weak Gaussian prior pulling
    weights toward 0, needed because a state whose effective (posterior-
    weighted) sample happens to be near-separable in contrast otherwise drives
    that weight to an arbitrarily large, poorly-identified value (see
    `fit_glm_hmm`'s docstring for how this was actually found, not assumed).
    l2=0 recovers the original unregularised fit.
    """
    def nll(w: np.ndarray) -> float:
        p = np.clip(_state_logistic_prob(contrast, w), _EPS, 1 - _EPS)
        data_term = -np.sum(weight * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        return data_term + 0.5 * l2 * float(np.sum(w ** 2))

    res = optimize.minimize(nll, x0, method="L-BFGS-B")
    return res.x if res.success else x0


def label_engaged_state(weights: np.ndarray) -> int:
    """Index of the state with the steepest |contrast slope| -- the
    "does the stimulus actually drive this state's choices" signature used to
    call a state "engaged" rather than biased/disengaged.
    """
    return int(np.argmax(np.abs(weights[:, 1])))
