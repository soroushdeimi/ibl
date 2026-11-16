# IBL prior / history pipeline — analysis report

**Date:** 23 August 2026  
**Dataset:** International Brain Laboratory public 2024 Neuropixels release  
**Reference paper:** Findling, Hubert et al., *Brain-wide representations of prior information in mouse decision-making*, Nature (2025), doi:10.1038/s41586-025-09226-1  
**Code:** `ibl_pipeline/`

This document records what the pipeline does, what was changed in the last pass, and what the numbers say. It is written so that a reader new to the project can follow the claim.

---

## Summary

Mice in the IBL block task use both the block prior and their recent choice/reward history when they choose (Stage 1, held-out accuracy about 83%). Pre-stimulus activity in CP / MOp / MOs / ORBvl, on raw counts, decodes the Bayes-optimal prior better than the pseudosession null; but after behavioural history is removed (`residuals`) the signal drops to zero in **all** regions. A behaviour-only decoder (`confounds`) is stronger than any neural decoder.

Two fixes in this pass: (1) the `and` bug in `n_test_sessions`, which did not affect any Stage 1 number; (2) the window was changed from −400…0 ms to Findling's −600…−100 ms. 84 sessions were re-extracted and `prior_decode` was re-run. The scientific story did not change. `reversal` has not been re-run after the window change.

Supported claim: in these eight regions, with this window and a linear decoder, no prior signal separable from action history was found. Unsupported claim: "the Findling result is an artefact". They tested 242 regions with 1,000 pseudosessions and report the dependence on past actions as a separate analysis, not inside the significance test.

---

## 0. One-sentence claim

Mice in the IBL block task clearly use both the block prior and recent choice/reward history when they choose. Pre-stimulus population activity in a handful of fronto-striatal regions *looks* like it encodes the Bayes-optimal prior — but that signal disappears once recent action history is removed, and a decoder that never sees a spike outperforms the neural decoder. Matching Findling et al.’s −600 to −100 ms window does not change this pattern.

---

## 1. What we did in this pass

Two reviewer-facing issues were fixed, then the affected stages were re-run.

### 1.1 Bug: `n_test_sessions` used Python `and`

In `fit_history_glm_nested_cv` (`ibl_pipeline/behavior_models.py`) the fold metadata was:

```text
n_test_sessions = g_train.nunique() and groups.iloc[test_idx].nunique()
```

Python `and` is not bitwise. If the left side is truthy it returns the right side. Here `g_train.nunique()` was always > 0, so the stored number happened to be the test-fold count. The line is now:

```text
n_test_sessions = int(groups.iloc[test_idx].nunique())
```

Stage 1 Model 1 was re-run with `--only model1` (GLM-HMM and belief models were *not* overwritten). `model1_cv_folds.csv` is bit-identical to the pre-fix backup. Every fold still has `n_test_sessions = 16`. Coefficients, log-likelihood and accuracy did not move.

### 1.2 Protocol match: prestimulus window

Findling et al. decode in the inter-trial interval with a main window of **−600 to −100 ms** relative to stimulus onset (Methods; they also report a narrower −400 to −100 ms check). This project had used **−400 to 0 ms**.

Changes:

- `config.PRESTIM_WINDOW_S = (-0.600, -0.100)`
- `extract.SCHEMA_VERSION` bumped from 1 to 2 so old count files are not treated as complete
- 84 sessions re-extracted (`schema_version: 2`, window `[-0.6, -0.1]`)
- `prior_decode` re-run on the new counts (200 pseudosessions per session×region)

A backup of the old summary is kept at `data/survey/stage2/prior_decode_summary_by_region_window400ms.csv`.

`reversal.py` was **not** re-run after the window change. Stage 3 numbers below are from the old −400…0 extract and are labelled as such.

---

## 2. What the pipeline does

The task is IBL `ephysChoiceWorld`: a visual grating appears left or right; the mouse reports the side by turning a wheel. Side probability (`probabilityLeft`) switches between 0.2 and 0.8 in blocks of variable length (mean ~50 trials), with an uncued 0.5 block at the start. A normative observer can track P(next stimulus on the right) with a forward filter; that quantity is the **Bayes-optimal prior**. Findling et al. report that this prior is decodable from 20–30% of brain regions across the sensory–motor hierarchy.

This project asks a narrower, control-heavy version of the same question in eight regions, and adds the control the paper reports beside the test rather than inside it: is the decoded “prior” separable from the animal’s own recent choices and rewards?

### 2.1 Regions

Chosen to sit on the same hierarchy Findling et al. highlight, plus CA1 (repeated-site coverage; latent-state / context inference):

| Acronym | Region | Level |
|---|---|---|
| LGd | dorsal lateral geniculate | early sensory (thalamus) |
| VISp | primary visual cortex | early sensory (cortex) |
| MOs | secondary motor cortex | motor preparation |
| MOp | primary motor cortex | motor preparation |
| ACAd | dorsal anterior cingulate | high-level associative |
| ORBvl | ventrolateral orbitofrontal | high-level associative |
| CP | caudoputamen | decision / action |
| CA1 | hippocampus CA1 | latent state / context |

Region membership uses the Allen CCF hierarchy (layers and subnuclei), not string equality.

### 2.2 Session filter

Only block-structured `biased` / `ephys` protocols. Training sessions (no real block prior) are dropped. Protocols that contain `ephys` but are not the standard task are also dropped: `_bandit_`, `_opto_`, `passive_`.

Survey coverage in the public release (insertions that pass the filter and have good units): LGd 47 sessions / 42 subjects; VISp 97 / 70; MOs 117 / 78; MOp 53 / 40; ACAd 69 / 57; ORBvl 13 / 11; CP 107 / 76; CA1 247 / 164.

The decode set is smaller: 84 extracted sessions, of which 70 contributed at least one region after unit/trial cuts (3 good units, 150 finite trials).

### 2.3 Stage 0 — extract

`python -m ibl_pipeline.extract`

Per session under `data/sessions/<eid>/`:

- trials and history covariates (lags 1–5)
- QC-passing units (`clusters.metrics` label == 1)
- pre-stimulus spike counts in `PRESTIM_WINDOW_S`
- a manifest with `schema_version` and the window actually used

Raw spike trains are pruned after counts are written. After this pass: **84/84** manifests, **78** `complete`, **6** `complete_no_units`, all schema 2, all window (−0.6, −0.1).

### 2.4 Stage 1 — behaviour (no spikes)

`python -m ibl_pipeline.run_stage1`

Pilot: 80 sessions, 50 subjects, 12 labs; 53,161 trials (52,791 valid; 45,961 in biased blocks).

**Model 1 — history GLM.** Logistic regression of choice on signed contrast, the block prior, and reward-gated choice history at lags 1–5. Nested session-grouped 5-fold CV for prediction; an unregularised cluster-robust fit for readable coefficients.

**GLM-HMM (3 states)** and **Model 2 (belief models)** were fit on an earlier run and were not refreshed in this pass.

### 2.5 Stage 2 — prior decode (the headline test)

`python -m ibl_pipeline.prior_decode`

Target: Bayes-optimal prior from a forward filter with hazard **h = 0.02** (1 / measured mean block length of 50). This is the paper’s target, not raw `probabilityLeft`.

Null: 200 pseudosessions that regenerate blocks → stimuli → belief from the rig’s generative process, scored on the **same** neural design matrix and the **same** interleaved folds. p is the fraction of null R² ≥ real R². Region-level p-values are combined with Fisher’s method and FDR-corrected (Benjamini–Hochberg) across the eight regions.

Three feature sets, same target and same null:

| Feature set | What is decoded | How to read it |
|---|---|---|
| `counts` | raw pre-stimulus spike counts | upper bound (history still in the neurons) |
| `residuals` | counts after regressing out choice/reward history (lags 1–2 plus interaction) | the claim: prior leftover after history |
| `confounds` | that history alone, no neural data | whatever this scores is available without a brain |

Findling et al. regenerate the stimulus sequence but do not simulate choices, so real action history stays in the neural data while the counterfactual prior is unrelated to it. Their significance test therefore does not separate prior coding from past-action coding. They report the dependence separately (orthogonalisation; correct-vs-error asymmetry). This pipeline puts that check inside the test.

### 2.6 Stage 3 — reversal (not re-run after the window fix)

`reversal.py` asks whether the first trial after a 0.2 ↔ 0.8 switch is decodable. That is the one place prior and history point in opposite directions. Those numbers below are from the old window.

---

## 2.7 Methodology by stage: goal, algorithm, what we ran

### Conventions (`config.py`, `one_client.py`)

**Goal.** One set of analysis choices for every session.  
**Algorithm / contract.** Eight CCF regions (Findling hierarchy + CA1); only `biased`/`ephys`; drop bandit, opto, passive; QC `label == 1`; region membership by Allen descendants, not string match; window −600 to −100 ms (schema 2); history lags 1–5. ONE talks to the public Alyx mirror without prompting.

### Survey (`insertions.py`, `survey.py`, `run_survey.py`)

**Goal.** Know coverage and power *before* downloading spike trains.  
**Algorithm.** Alyx REST search by acronym; each insertion gets a protocol verdict *with a reason*; cache `clusters.*` + trial counts only.  
**What we ran.** Full public 2024 release. Example: CP 107 sessions / 76 subjects; ORBvl 13 / 11. Underpowered flag: <3 subjects or <5 sessions.

### Extract (`behavior.py`, `spikes.py`, `covariates.py`, `extract.py`)

**Goal.** One on-disk schema: trials, QC units, prestim counts, history, aligned row-for-row.  
**Algorithms.** Trials: IBL choice sign (−1 = right); `prior_bias_right = 0.5 − pL`; reward-gated vs unrewarded history at lags 1–5; omissions dropped before shift. Spikes: peak-channel CCF id; `searchsorted` counts in `PRESTIM_WINDOW_S`; no QC label → refuse the insertion. Covariates: wheel (1 kHz), whisker ME, pupil, same window.  
**What we ran.** 84 sessions, schema 2; 78 complete, 6 complete_no_units; spike trains pruned after counts. Decode cuts: ≥3 units, ≥150 trials → 70 sessions, 110 pairs, 49 mice.  
`sanity_check.py` locks window edges, alignment, and wheel-velocity sign so an off-by-one cannot look like a result.

### Stage 1 (`behavior_models.py`, `run_stage1.py`)

**Goal.** Show, without spikes, that mice use *both* block prior and choice/reward history — the confound Stage 2 must beat.

| Algorithm | What it is | What we ran / why |
|---|---|---|
| Model 1 history GLM | Logistic choice ~ contrast + block prior + (r,u) lags 1–5. Nested **session-grouped** 5-fold CV (inner loop only picks C). Unregularised cluster-robust fit for coefficients. | 80 sessions, 50 mice. Accuracy 82.7%, pseudo-R² 0.433. History still significant at lag 5. `and`-bug in `n_test_sessions` fixed; numbers unchanged. |
| 3-state GLM-HMM | Engaged vs left/right-biased Bernoulli GLM. | Labels engaged trials for Model 2 only — not a neural filter. Engaged: near-zero bias, steepest slope (38% of trials). Not re-fit in this pass. |
| Model 2 belief models | Bayes-optimal / fixed-rate / two-rate / surprise-scaled. Per-session MLE; **temporal** hold-out (shuffle would leak through the recursion). `a_t ∈ {0,1}` so `π_t` stays in (0,1). | On engaged trials Bayes-optimal wins; on all trials fixed-rate wins. Not re-fit in this pass. |

### Stage 2 generations (only v4 is the claim)

| Gen | Module | Goal | Algorithm | What happened |
|---|---|---|---|---|
| v1 | `decode.py` | Decode Stage 1 `π̂` from counts | Engaged-only; contiguous time-block CV; R² vs 0 | Negative R² everywhere. Causes: degenerate `π̂` (hazard MLE at 0.5); N cut ~60%; baseline on those folds is R² = −0.54 |
| — | `stage2_controls.py` | Is the pipeline blind? | Same ridge vs trial index, prev choice, block, `π̂` | Trial index R² 0.64; block above chance in 95% of pairs; `π̂` still ~0.02. Pipeline OK; target/CV not. Smoke test only |
| v2 | `pseudosessions.py` + `stage2_null.py` | p-value that includes drift + autocorrelation | Fake blocks from the rig (90-trial pL=0.5, then 0.2/0.8, truncated exp [20,100]); real neurons; shared interleaved folds; each target picks its own ridge α | Decoded raw `probabilityLeft`, still with history inside the neurons |
| v3 | `pseudopopulation.py` | Is a region just underpowered? | Pool units across sessions (CP 51→770). Null *rebuilds* the population; permuting labels would kill the drift confound | Behaviour-only AUC 0.985–1.000. Pooling amplified history, not prior |
| **v4** | **`prior_decode.py`** | **Match Findling, and put history inside the test** | **Bayes-optimal prior, h=0.02 fixed; window −600…−100; 200 pseudosessions; Fisher+FDR. Three feature sets: counts / residuals / confounds** | **Headline. counts: CP/MOp/MOs/ORBvl survive FDR. residuals: null everywhere. confounds: excess R² 0.55–0.67** |

Findling regenerates stimuli but not choices, so real action history stays in the neurons while the fake prior is unrelated to it. They report that dependence separately. We scored it as a third decoder.

### Stage 3 (`reversal.py`)

**Goal.** The only moment prior and history oppose: first trial after 0.2 ↔ 0.8.  
**Algorithm.** Train on steady state (lag ≥ 10), test after reversal. AUC > 0.5 = new block, < 0.5 = old. Primary test lag 1 (narrowed after the behaviour-only curve, not after neural results). Opening 0.5→biased is not a reversal.  
**What we ran.** Behaviour-only AUC 0.10–0.16 at lag 1 (design works). Neural counts in CP/MOp beat the null; residuals do not. **Not re-run after the window change.**

---

## 2.8 The formulas that actually score

The working idea is one sentence: the next choice is contrast + block prior + reward-gated action history. Stage 2 applies that same idea to spikes. The Bayes filter is only the *target*, not what explains the neural numbers.

### Choice (Stage 1)

\[
P(a_t=\mathrm{right})
=
\sigma\!\big(
\beta_0
+ \beta_c\,c_t
+ \beta_\pi\,\pi^{\mathrm{block}}_t
+ \sum_{k=1}^{5}
  (\beta_k^{r}\,r_{t-k}+\beta_k^{u}\,u_{t-k})
\big)
\]

\(\sigma(x)=1/(1+e^{-x})\). Signed contrast is \(+\)contrast on a right stimulus, \(-\)contrast on a left one. Block term: \(\pi^{\mathrm{block}}_t=0.5-p_L\) (a right-favouring block \(p_L=0.2\) is \(+0.3\)).

History, in `behavior.py`:

\[
s_{t-k}=2a_{t-k}-1\in\{-1,+1\},\qquad
r_{t-k}=s_{t-k}\cdot\mathbf{1}_{\mathrm{rewarded}},\qquad
u_{t-k}=s_{t-k}\cdot\mathbf{1}_{\mathrm{unrewarded}}
\]

Every history coefficient is positive through lag 5. Held-out accuracy 82.7%. This does **not** kill the prior (\(\beta_\pi\approx +0.99\)). It says the mouse uses both — that is the confound.

### The same formula on spikes (Stage 2)

Per unit, OLS on that session:

\[
\mathrm{counts}_t
=
B_0+B_1 a_{t-1}+B_2 R_{t-1}+B_3 a_{t-2}+B_4 R_{t-2}+B_5(a_{t-1}R_{t-1})+\varepsilon_t
\]

\[
\mathrm{residuals}_t=\mathrm{counts}_t-\widehat{\mathrm{counts}}_t
\]

Ridge then decodes the Bayes-optimal prior from \(\varepsilon\). A separable prior would remain in \(\varepsilon\). It does not: excess \(R^2\) is negative everywhere. `confounds` is the right-hand side with no spikes (excess \(R^2\approx 0.55\)–\(0.67\)). History without a brain outruns every neural decoder because prior and recent actions are nearly the same variable in this task.

Residualisation is fit once per session, not inside CV. Regressors are history only, never the block label, so the target cannot leak. The test can only understate leftover prior coding, never invent it.

### The decode target (Bayes filter, \(h=0.02\))

Two hidden states, \(P(\mathrm{right})=0.8\) or \(0.2\). Entering trial \(t\):

\[
P(\mathrm{high}_t)=(1-h)\,P(\mathrm{high}_{t-1}\mid\mathrm{data})+h\,P(\mathrm{low}_{t-1}\mid\mathrm{data})
\]

\[
\pi_t=0.8\,P(\mathrm{high}_t)+0.2\,P(\mathrm{low}_t)
\]

Then Bayes update on \(P(\mathrm{high})\) after the stimulus. This \(\pi_t\) is Findling’s target and ours. The formula is correct. The problem is collinearity with \((a_{t-1},R_{t-1},\ldots)\), so decoding \(\pi_t\) from raw counts does not by itself mean the population holds a prior.

---

## 3. Results

### 3.1 Stage 1 — mice use both prior and history

Unregularised Model 1, N = 52,391 trials, cluster-robust SEs, pseudo-R² = 0.433.

| Predictor | Coefficient | p |
|---|---|---|
| signed contrast | +4.98 | < 10⁻²⁴ |
| block prior (right) | +0.99 | < 10⁻³⁵ |
| rewarded choice, lag 1 | +0.45 | < 10⁻⁶⁹ |
| unrewarded choice, lag 1 | +0.49 | < 10⁻²⁷ |
| rewarded choice, lag 5 | +0.12 | < 10⁻¹² |
| unrewarded choice, lag 5 | +0.40 | < 10⁻¹⁸ |

Held-out (session-grouped nested CV): mean accuracy **82.7%**, mean log-likelihood per trial **−0.394**, 16 test sessions per fold.

So behaviour is not “prior or history”. It is both, and history is still detectable five trials back. That is the confound Stage 2 has to beat.

Belief models on the earlier engaged-state pass (79 sessions, not re-fit this week): Bayes-optimal held-out LL slightly better than a fixed-rate kernel (−0.297 vs −0.304). That is a behavioural fact, not a neural one.

### 3.2 Stage 2 — counts look like a prior; residuals do not

Decode set: 110 session×region pairs, 70 sessions, 49 subjects. Window −600 to −100 ms. 200 pseudosessions.

**Neural counts (upper bound)**

| Region | Sessions | Subjects | Median units | Excess R² | n sig. | Fisher p | BH q | Survives FDR? |
|---|---|---|---|---|---|---|---|---|
| CP | 15 | 12 | 51 | +0.047 | 7 | 7.9×10⁻⁶ | 6.3×10⁻⁵ | yes |
| MOp | 5 | 4 | 21 | +0.042 | 2 | 0.0030 | 0.012 | yes |
| ORBvl | 1 | 1 | 84 | +0.110 | 1 | 0.015 | 0.040 | yes (underpowered) |
| MOs | 18 | 14 | 18 | +0.018 | 2 | 0.020 | 0.040 | yes |
| LGd | 13 | 12 | 10 | 0.000 | 0 | 0.55 | 0.84 | no |
| ACAd | 9 | 8 | 16 | 0.000 | 0 | 0.63 | 0.84 | no |
| VISp | 17 | 13 | 12 | −0.006 | 0 | 0.82 | 0.87 | no |
| CA1 | 32 | 28 | 10 | 0.000 | 0 | 0.87 | 0.87 | no |

On raw counts, the same four regions Findling et al. highlight on the frontal–striatal side (CP, MOp, MOs, ORBvl) beat the pseudosession null after FDR. Early sensory (LGd, VISp), ACAd and CA1 do not.

**Residuals (the claim)**

Every region: `n_sig_residuals = 0`, excess R² negative (about −0.01 to −0.06), BH q ≈ 1.

After choice and reward history (lags 1–2) are partialled out of the counts, nothing decodable remains. The “prior” in the counts was the history the mouse just produced.

**Confounds (behaviour only)**

Identical before and after the window change, as they must be (no spikes):

- excess R² ≈ 0.55–0.67
- every session p = 0.005 (floor with 200 nulls)
- Fisher p from 10⁻⁸ (MOp) to 10⁻³⁹ (CA1)

A decoder that never sees a neuron is stronger than any neural decoder in this set. That is the result that should travel with the figure.

### 3.3 Window comparison: −400…0 vs Findling −600…−100

| Region | Excess R² old → new | n sig. old → new | BH q old → new |
|---|---|---|---|
| CP | +0.043 → +0.047 | 5 → 7 | 5.3×10⁻⁴ → 6.3×10⁻⁵ |
| MOp | +0.025 → +0.042 | 2 → 2 | 0.012 → 0.012 |
| MOs | +0.028 → +0.018 | 2 → 2 | 0.030 → 0.041 |
| ORBvl | +0.110 → +0.110 | 1 → 1 | 0.040 → 0.040 |
| LGd / ACAd / VISp / CA1 | ≈ 0 | 0 → 0 | all > 0.57 |

Residuals stayed null in both windows. Confound scores did not change.

Matching the paper’s window slightly *strengthened* CP (two more sessions over the uncorrected 0.05 line) and MOp excess R², and slightly *weakened* MOs. The scientific story is the same.

That direction is expected if the counts are mostly an echo of the previous trial’s movement and reward: the Findling window starts 200 ms earlier and ends 100 ms before onset, so it sits farther from stimulus edge and closer to the last action. Residualising still kills the signal.

### 3.4 Stage 3 — reversal (old window; not re-run)

On the first trial after a block switch, history and the new prior disagree. Under the old −400…0 counts, raw-count AUC in CP (0.36) and MOp (0.28) beat the null (BH q = 0.035), but residual AUC did not (p = 0.26 and 0.52). Confound AUC on that same first trial was *below* 0.5 and significant — history still points at the *old* block. This is consistent with Stage 2: the neural “prior” tracks the last action, not the hidden state. These numbers should be recomputed on schema-2 extracts before they are cited as current.

---

## 4. How to read this against Findling et al.

What this project **matched**:

- target = Bayes-optimal prior, not raw `probabilityLeft`
- window = −600 to −100 ms
- per-session linear decode, interleaved CV, R², pseudosession null, Fisher combination

What it **did not match** (and should not pretend to):

- 8 regions vs 242
- 200 pseudosessions vs 1,000 in their main test
- no wheel-stillness / ITI movement screen
- no widefield imaging arm
- residualisation of history is *inside* the test here; in the paper it is a separate analysis

A negative residual result in these eight regions, with a linear decoder, does **not** license “the Findling result is an artefact.” It licenses: “in these regions, in this window, with this decoder, no prior signal separable from action history was found.” The paper itself reports that past actions influence the decoded prior for at least five trials and that a correct-vs-error pattern fits an action kernel better than a Bayes-optimal observer. The disagreement is about whether that dependence is a side note or the whole signal in the regions we can power.

ORBvl is one session. ACAd is powered and null on counts. LGd and VISp are powered and null on counts. That last point is the cleanest contrast with the paper’s “early sensory too” claim, and it survives the window alignment.

---

## 5. Files

| Path | Contents |
|---|---|
| `ibl_pipeline/config.py` | regions, window, QC, paths |
| `ibl_pipeline/extract.py` | schema 2 extract |
| `ibl_pipeline/run_stage1.py` | behaviour models |
| `ibl_pipeline/prior_decode.py` | Stage 2 headline test |
| `data/survey/stage1/model1_*.csv` | history GLM |
| `data/survey/stage2/prior_decode_summary_by_region.csv` | current (Findling window) |
| `data/survey/stage2/prior_decode_summary_by_region_window400ms.csv` | backup (−400…0) |
| `data/survey/stage3/reversal_summary.csv` | old window; do not cite as current |

Commands used in this pass:

```text
py -3 -m ibl_pipeline.run_stage1 --only model1
py -3 -m ibl_pipeline.extract --eids <84 eids> --prune-cache
py -3 -m ibl_pipeline.prior_decode
```
