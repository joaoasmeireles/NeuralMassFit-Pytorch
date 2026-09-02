# BPKF-Torch — GPU-Accelerated Neural-Mass Model Inversion for EEG

> **Provisional name.** This project is an independently developed,
> GPU-accelerated implementation of the MINDy-style neural-mass inversion via
> Backpropagation through a Kalman Filter (BPKF). It is **not affiliated with or
> endorsed by** the original MINDy authors. Pending direct communication with
> them, we avoid using the "MINDy" name for this release. Rename freely to suit
> your project.

A PyTorch, GPU-accelerated pipeline that fits a mesoscale neural-mass model to
source-space EEG, one sliding window at a time, and extracts per-window
effective-connectivity matrices and dynamical summaries. It was developed for
the study *[YOUR PAPER TITLE]* ([YOUR VENUE/YEAR]) and is released to support
reproducibility.

---

## Attribution and license

This implementation derives from and extends the MINDy-BPKF method and reference
implementation of **Matthew Singh and colleagues**, released under the MIT
License (Copyright (c) 2024 Matthew Singh). In accordance with that license, the
original copyright notice and permission notice are retained in [`LICENSE`](LICENSE).

Our contribution is a source-space adaptation with GPU execution and a set of
stability/regularization changes ("anti-hub / anti-idiosyncrasy" hardening);
these are documented in the header of `fit_mindy_pytorch_sourceSpace.py`. This
work has **not** been reviewed or endorsed by the original authors.

If you use this code, please cite both:

- **This work:** [YOUR CITATION]
- **The original method:** M. Singh et al., *[MINDy / BPKF reference]*.

---

## What it does

Given source-projected EEG (per subject, session, and condition), the pipeline:

1. Slides a 30 s window (50% overlap, 250 Hz) across each recording.
2. Fits a continuous-time recurrent neural-mass model per window by
   Backpropagation through a Kalman Filter (BPKF), using an anatomical
   lead-field `H` as the fixed measurement operator.
3. Saves, per window, the excitatory effective-connectivity matrix `W_EE`, the
   recurrent gain `S`, noise terms, and derived scalar/dynamical features.

The model is a mask-structured E/I network (68 excitatory + 68 inhibitory
populations for a 68-region Desikan–Killiany parcellation). Connectivity is the
learned parameter set; see the paper for the interpretation and its limits
(the fitted parameters are model-derived, not causally validated).

---

## Installation

Requires Python 3.9+, a CUDA-capable GPU (optional; falls back to CPU), and:

```bash
pip install torch numpy scipy matplotlib
```

`torch` should be installed with the CUDA build matching your system
(see https://pytorch.org). Fitting is substantially faster on GPU
(~50 min per subject-session on an RTX 3080 Laptop; far longer on CPU).

---

## Input format

The script reads MATLAB `.mat` files, one per subject/session/condition, from
`MINDy_source/sub-XX/`. Each file must contain:

| key             | shape            | description                                 |
|-----------------|------------------|---------------------------------------------|
| `MeasData`      | `(n_channels, T)`| source-projected (or sensor) EEG signal     |
| `H`             | `(n_channels, n_pop)` | lead-field / measurement operator      |
| `n_channels`    | scalar           | number of channels                          |
| `n_populations` | scalar           | total populations (`n_regions × (exc+inh)`) |
| `n_exc_per_ch`  | scalar           | excitatory populations per region           |
| `n_inh_per_ch`  | scalar           | inhibitory populations per region           |
| `channel_names` | list             | channel labels                              |
| `n_regions`     | scalar           | number of cortical regions (e.g. 68)        |
| `region_names`  | list             | region labels (recommended)                 |

> The forward model `H` and the preprocessing (resampling, band-pass, CAR) are
> produced by a separate forward-model script; this repository covers the
> inversion step. A minimal synthetic example is provided in
> [`examples/demo.ipynb`](examples/demo.ipynb) so the pipeline can be run without
> the original dataset.

---

## Usage

```bash
# fit all subjects, auto-select GPU/CPU
python fit_mindy_pytorch_sourceSpace.py --subject all --device auto

# fit a subset
python fit_mindy_pytorch_sourceSpace.py --subject sub-01,sub-02 --device cuda
```

**Flags**

| flag        | default | description                                   |
|-------------|---------|-----------------------------------------------|
| `--subject` | `all`   | `all`, or a comma-separated list of subjects  |
| `--device`  | `auto`  | `auto`, `cuda`, or `cpu`                       |

Key constants (window length, overlap, sampling rate, fixed decay) are defined
near the top of the script.

**Checkpointing.** The script skips any subject whose `window_features.mat`
already exists, so runs can be stopped and resumed. Saving occurs once per
subject, at the end of that subject's fit — interrupting mid-subject loses only
the in-progress subject.

---

## Output format

For each subject, `results_source/sub-XX/window_features.mat` contains, stacked
over all windows:

| key           | description                                    |
|---------------|------------------------------------------------|
| `allW`        | effective-connectivity matrices per window     |
| `allS`        | recurrent gain per window                       |
| `allR_diag`   | measurement-noise terms                         |
| `allFeatures` | derived scalar features                         |
| `allLabels`   | condition label per window (0/1/2-back)         |
| `allSessions` | session index per window                        |
| `allWindows`  | within-recording window index                  |

---

## Data

This repository does **not** include EEG data. The study used the COG-BCI
multi-session corpus (Hinss et al.), available at its original DOI
[10.5281/zenodo.6874128]. The synthetic demo requires no external data.

---

## Reproducibility notes

- The exact window length used in the published study is `WINDOW_SEC = 30`.
- Fitting is stochastic in initialization; per-window seeds are set where noted.
- The operating-point Jacobian analyses use an approximation (`x ≈ 0`) because
  the offset/input terms are not stored; see the paper for the implications.

---

## License

MIT. See [`LICENSE`](LICENSE). The original MINDy copyright and permission
notice (Copyright (c) 2024 Matthew Singh) are retained therein as required.
