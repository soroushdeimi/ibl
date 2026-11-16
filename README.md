# IBL prior decoding

A re-analysis, with added controls, of block-prior coding in the International Brain Laboratory (IBL) public Neuropixels release.

Findling, Hubert et al. (*Brain-wide representations of prior information in mouse decision-making*, Nature 2025, doi:10.1038/s41586-025-09226-1) report that the Bayes-optimal block prior can be decoded from pre-stimulus activity in many brain regions. This project asks a narrower question in eight regions: is the decoded prior separable from the animal's own recent choices and rewards?

## Result in one paragraph

Mice use both the block prior and their recent choice and reward history when they choose (held-out accuracy about 83%). On raw spike counts, pre-stimulus activity in CP, MOp, MOs and ORBvl decodes the prior better than a pseudosession null. After recent action history is regressed out, the signal is gone in every region, and a decoder that uses behaviour only is stronger than any neural decoder. In these eight regions, with the −600 to −100 ms window and a linear decoder, no prior signal separable from action history was found. This does **not** show that the published result is an artefact: the paper covers 242 regions with 1,000 pseudosessions and reports the dependence on past actions as a separate analysis. Details, limits and the list of what was and was not matched are in [`docs/analysis_report.md`](docs/analysis_report.md).

## Regions

LGd, VISp (early sensory), MOs, MOp (motor preparation), ACAd, ORBvl (associative), CP (striatum), and CA1 (hippocampus), chosen to cover the sensory-to-motor hierarchy. See `ibl_pipeline/config.py`.

## Pipeline

| Stage | What it does | Entry point |
|---|---|---|
| Survey | Find insertions and sessions that cover the target regions, apply QC | `python -m ibl_pipeline.run_survey` |
| Extract | Trials, clusters, behavioural covariates and pre-stimulus spike counts per session | `python -m ibl_pipeline.extract` |
| Stage 1 | Behaviour models: history GLM, 3-state GLM-HMM, and four belief models (Bayes-optimal and three action-based) | `python -m ibl_pipeline.run_stage1` |
| Stage 2 | Prior decoding per session and region against a pseudosession null, on counts and on history residuals | `python -m ibl_pipeline.prior_decode` |
| Stage 3 | Population time course around block reversals | `python -m ibl_pipeline.reversal` |
| Report | Data-pipeline summary | `python -m ibl_pipeline.report` |

Stage 3 has not been re-run since the window change; its numbers come from the earlier −400 to 0 ms extract.

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from one.api import ONE; ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)"
```

Data comes from the public IBL server (OpenAlyx) through the ONE API. No credentials are stored in this repository.

## Repository layout

| Path | Contents |
|---|---|
| `ibl_pipeline/` | Pipeline code |
| `data/survey/` | Survey tables and stage outputs (Stage 1 to 3 summaries) |
| `data/sessions/` | Per-session extracts |
| `docs/analysis_report.md` | Methods, results and how to read them against the paper |
| `docs/data_pipeline.md` | Data acquisition and QC |
| `docs/latex/` | Written report and figures |
| `IBL_prior_choice_exploration.ipynb` | First behavioural exploration (one subject) |
| `connection_check.ipynb` | Configuration and server connection check |
