# AGENTS.md — MyEEG

Primary guidance for AI coding agents (Cursor, Claude, etc.) working in this repository. `CLAUDE.md`-style tools should treat this file as the source of truth.

## 1. Project vision

MyEEG is a **personalized EEG electrode-selection and placement optimization system** for brain-computer interfaces (BCIs).

The goal is to find out which EEG channels / scalp positions carry the most useful information for decoding a *particular user's* intended action, and to test whether personalized configurations can reach comparable or better decoding with **fewer electrodes, less calibration time, or better robustness**.

| Level | Scope | Status |
| --- | --- | --- |
| **1. Software channel selection** | Choose a channel subset from a fixed, already-recorded montage. | **Current priority** |
| 2. Personalized placement recommendations | Recommend physical electrode positions using anatomy, task, and signal quality. | Future research |
| 3. Reconfigurable hardware | Hardware that can enable/disable or move electrodes. | Future research |

Do not start Level 2 or 3 work until Level 1 shows a measurable result on real EEG.

> Previous direction: the repo began as a "focus / distraction detector + contextual recommendation agent." That idea is now an **optional extension** (see [§9](#9-existing-and-earlier-components)).

## 2. Core research question

> Can a personalized, task-aware electrode-selection optimizer decode intended movements accurately using fewer EEG channels than a fixed-channel baseline?

Metrics to report:

- Held-out classification performance (e.g., accuracy, balanced accuracy, Cohen's kappa).
- Number of channels used.
- Amount of calibration data / time needed to reach a given performance.
- Robustness to noisy or dropped channels.
- Cross-session performance, **only where the data actually contains multiple sessions**.

**Do not assume the optimizer beats the full-channel baseline.** That is the empirical question. A negative result ("8 channels lose 10 points of accuracy") is a valid, reportable result.

## 3. Initial BCI task and dataset

Task: **motor-imagery classification** — start with **binary left-hand vs right-hand imagery**, then add rest if it is cleanly supported.

Candidate dataset: **PhysioNet EEG Motor Movement/Imagery Dataset (`eegmmidb`, v1.0.0)**, also loadable via `mne.datasets.eegbci`.

Verified facts (checked 2026-10-09 against the [PhysioNet page](https://physionet.org/content/eegmmidb/1.0.0/) and the [MNE `eegbci.load_data` docs](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html)):

- 109 subjects (folders `S001`–`S109`; MNE accepts subjects 1–109).
- 64 EEG channels, international 10-10 system (excluding Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, P10), sampled at **160 Hz**, EDF+ format with an annotation channel.
- 14 runs per subject:

| Run(s) | Task |
| --- | --- |
| 1 | Baseline, eyes open (1 min) |
| 2 | Baseline, eyes closed (1 min) |
| 3, 7, 11 | Motor **execution**: left vs right fist |
| 4, 8, 12 | Motor **imagery**: left vs right fist |
| 5, 9, 13 | Motor execution: both fists vs both feet |
| 6, 10, 14 | Motor imagery: both fists vs both feet |

- Annotations: `T0` = rest; `T1` = left fist (runs 3/4/7/8/11/12) **or** both fists (runs 5/6/9/10/13/14); `T2` = right fist **or** both feet. **The meaning of T1/T2 depends on the run number** — always map labels using the run.
- Signal indices in files are 0–63; the montage figure numbers electrodes 1–64.
- `mne.datasets.eegbci.standardize(raw)` standardizes channel names and positions (the raw EDF channel names are not standard 10-10 labels).

So for left vs right imagery, use **runs 4, 8, 12** and the T1/T2 events.

**Not yet verified — check in code before relying on it:**

- Whether every subject has all 14 runs at full length and 160 Hz. Some papers report excluding a few subjects for irregular recordings; confirm per subject by loading the data.
- Whether all runs were recorded in a single sitting. The documentation does not describe multiple sessions, so **assume within-session only** until proven otherwise. True cross-session evaluation will likely need a different dataset (e.g., BCI Competition IV 2a, which has two sessions per subject — also unverified here).
- Exact trial durations and counts per run (inspect the annotations).

General rule: do not assume every recording has 64 usable channels, or that electrode names/locations match across datasets. Validate channels per recording (pipeline step 3).

## 4. Pipeline

Build incrementally, one stage at a time, each with a small test or experiment:

1. **Acquisition / dataset loading** — download a few subjects, load EDF, map annotations to labels.
2. **Preprocessing & artifact handling** — filtering (e.g., 8–30 Hz band-pass for mu/beta), epoching, bad-channel / bad-epoch handling.
3. **Channel & electrode-location validation** — standardize names, attach a montage, flag flat/noisy channels, confirm the eligible channel set.
4. **Feature extraction & baseline classification** — e.g., log band power or CSP + LDA / logistic regression on all eligible channels.
5. **Channel subset optimization** — select subsets (see [§5](#5-baseline-before-optimization)).
6. **Evaluation on unseen trials / sessions** — see [§6](#6-evaluation-and-rigor).
7. **Personalized configuration recommendations** — per-subject report: chosen channels, performance vs channel count, stability of the choice.
8. **(Optional) Real-time inference & hardware** — only after steps 1–7 work offline.

Tools: Python, MNE-Python, NumPy/SciPy, scikit-learn. LSL / `pylsl` only when real-time acquisition is actually needed. Use the simplest tool that works for the current phase; do not add dependencies without explaining why.

## 5. Baseline before optimization

1. Build a **reproducible full-channel baseline** first (fixed random seeds, fixed splits, pinned dependency versions, documented subject/run selection).
2. Then compare channel subsets of size **8, 16, 32** (where the montage supports them), plus simple fixed references such as a standard motor-strip set (e.g., C3, Cz, C4 and neighbors).
3. Candidate selection methods — compare, do not assume a winner; start with the simplest:
   - Greedy forward / backward selection (wrapper around the classifier).
   - Feature-selection scores computed on training data only (e.g., mutual information, CSP filter weights).
   - Genetic algorithms.
   - Bayesian optimization — only if the simpler methods justify the extra complexity.
4. **Objective must be explicit.** Example starting point:

   `score(S) = validation_accuracy(S) − λ · |S|`

   where `S` is the channel subset and `λ` is a documented trade-off weight. Any additional penalty (setup difficulty, signal-quality cost) must be written down as a term with a defined unit and weight before it is used.

## 6. Evaluation and rigor

Leakage rules (non-negotiable):

- **Split the data before any channel selection.** Channel selection is part of model training.
- Fit preprocessing statistics, feature selection, channel selection, and model parameters on **training data only**. Use nested cross-validation (inner loop selects channels, outer loop evaluates).
- Evaluate on held-out trials; use separate sessions where the data has them.
- Personalization means **subject-specific** evaluation (train and test within the same subject).
- Always label which regime a number comes from: **within-session**, **cross-session**, or **cross-subject**. Do not mix them in one comparison.
- Always compare against (a) the full-channel baseline and (b) a simple selection baseline (e.g., fixed motor-strip channels, random subsets of the same size).
- Report metrics with uncertainty: mean ± std or confidence intervals across folds and subjects; note chance level (50% for balanced binary).

Interpretation rule: **selecting a channel does not prove that the brain region under it generated the signal.** Scalp EEG mixes activity through volume conduction; channel selection is not source localization.

## 7. Signal interpretation

- Alpha/beta band power is **not** automatically a reliable measure of focus, distraction, or intention. Use a defined task with labeled data, then train and evaluate a decoder.
- Keep four things distinct:
  1. Observed scalp potentials (what the electrodes measure).
  2. Estimated cortical source activity (a model-based inverse estimate).
  3. Predicted movement intention (a classifier output).
  4. Actual physical movement (ground truth only in execution runs).
- If source localization is added later (e.g., minimum-norm estimation, sLORETA via MNE), document its assumptions: head model, electrode positions (template vs measured), noise covariance, and limited spatial resolution with ~64 electrodes.

## 8. Learning approach (Build-and-Deconstruct)

Full blueprint: [`docs/learning-blueprint.md`](docs/learning-blueprint.md). Agents must:

- **Inspect the repo before proposing changes.**
- Explain the purpose of every module and every dependency before adding it.
- Teach **one concept at a time**; do not dump a whole implementation.
- Ask the developer to reason about decisions (e.g., "Why must we split before selecting channels?") before giving the answer.
- Explain the math (filters, band power, CSP, cross-validation) in plain language plus the equation.
- Encourage controlled experiments and intentional breaking ("what happens with 4 channels?").
- Add a test or small experiment for each major component.
- Never replace working code without explaining why.
- Clearly distinguish **implemented** vs **tested** vs **planned** (see [§11](#11-status)).

Environment: Windows + PowerShell, Python 3.12 in a `.venv` virtual environment (activate with `.\.venv\Scripts\Activate.ps1`). Give commands for PowerShell.

## 9. Existing and earlier components

These scripts came from the original blueprint and are now in the repo. Inspect them before changing them, and do not rewrite them without explaining why:

| File | What it is today | Maps to |
| --- | --- | --- |
| `streamer.py` | Working simulator: 4-channel, 256 Hz LSL stream (`fakeEEG`) of a 10 Hz sine with per-channel amplitude/phase and Gaussian noise. | Optional extension: real-time infrastructure testing (pipeline stage 8) |
| `detector.py` | Stub (TODO comments only): LSL inlet, FFT alpha/beta power, focus score. | Band-power code may later feed stage 4 (features); the focus score is an optional extension |
| `agent.py` | Stub (TODO comments only): contextual-bandit activity recommender. | Optional extension; no defined BCI objective yet |
| `requirements.txt` | `pylsl`, `numpy`, `mne`, `scikit-learn` (pinned). | scikit-learn added for the full-channel baseline |

General guidance:

- **EEG simulator / LSL streamer & receiver** — useful for **infrastructure testing only** (stream plumbing, buffering, real-time latency). Simulated EEG cannot establish real-world BCI performance.
- **Signal-processing code** (FFT, band power) — reusable for feature extraction.
- **Contextual recommendation agent / contextual bandit, user feedback, "focus score"** — optional extensions. Keep only if they serve a defined BCI objective; otherwise park them.
- **Visualizations** — reuse for topomaps, channel-importance plots, accuracy-vs-channel-count curves.

## 10. Hardware roadmap

No physical electrode-moving mechanism in the initial software phase. First establish a measurable benefit of channel selection on real EEG. If results support it, investigate:

- Standardized scalp coordinate systems (10-20 / 10-10) and reproducible placement.
- Contact impedance and signal quality.
- Individual anatomical differences (head size/shape).
- Mechanical repeatability and comfort.
- Whether the acquisition hardware can disable channels independently.
- Choosing among existing channels vs moving electrodes to unsampled locations (the latter cannot be validated from a fixed recording).

Any prototype must prioritize **electrical safety, isolation, and suitable acquisition hardware** (e.g., medical/research-grade isolated amplifiers; never connect electrodes on a person to mains-powered equipment without proper isolation).

## 11. Status

| Item | Status |
| --- | --- |
| Documentation (this file, README, learning blueprint) | Implemented |
| Dataset loading (eegmmidb) | Implemented, untested — `load_data.py` loads subject 1, runs 4/8/12, renames T0/T1/T2 per run before concatenating (checked by hand: 23 left / 22 right / 45 rest) |
| Preprocessing & channel validation | Implemented, untested — `preprocess.py`: `colin27_1005` montage, 8–30 Hz band-pass, robust-z bad-channel check on filtered data (threshold \|z\| > 5; subject 1 flags T10), epochs −0.5 to 4.0 s for left/right → shape (45, 63, 721) |
| Full-channel baseline | Implemented, untested — `baseline.py`: log band power (0.5–4.0 s) + L2 logistic regression, stratified 5-fold, seed 0. Subject 1, within-session: accuracy 0.533 ± 0.093 vs 0.511 majority-class (24/45 vs 23/45). CSP (6 filters, log-variance, fit inside each training fold) + logistic regression on the same folds: 0.778 ± 0.157. |
| Channel subset optimization | Implemented, untested — `select_channels.py`: greedy forward selection with CSP + LR, scored by inner 5-fold CV on training trials only; objective `accuracy − 0.005·|S|` picks a size along the greedy path. Subject 1, within-session, outer 5-fold: full 63ch 0.778 ± 0.157, greedy 8ch 0.644 ± 0.183, 16ch 0.711 ± 0.099, 32ch 0.689 ± 0.214, penalized (5–16ch) 0.667 ± 0.136. Chosen channels differ a lot between folds (no channel picked in more than 3/5 folds for the 8-channel set). ~9 min on 12 cores. |
| Evaluation framework (nested CV) | Implemented, untested — nested CV in `select_channels.py` (outer 5-fold seed 0, same folds as `baseline.py`). `fast_csp.py`: CSP on per-trial covariance matrices, checked to give identical fold scores to `mne.decoding.CSP`; greedy selection takes ~2.5 min per subject on one core. |
| Multi-subject comparison | Implemented, untested — `evaluate.py`: subjects 1–10, within-session, CSP + LR. Mean ± std across subjects: all channels 0.593 ± 0.230; 8ch greedy 0.564 ± 0.173 / motor strip 0.587 ± 0.197 / random 0.539 ± 0.104; 16ch 0.564 / 0.602 / 0.551; 32ch 0.587 / 0.589 / 0.581. Paired greedy − motor: −0.022 (8ch), −0.038 (16ch), −0.002 (32ch). Only subjects 1, 2, 7 decode clearly above chance with all channels. Figure: `results/accuracy_vs_channels.png`. ~12 min on 12 cores. |
| Personalized recommendations | Planned — `evaluate.py` prints each subject's greedy 8-channel picks, but they are unstable across folds |
| Real-time inference / LSL | Planned (optional) |
| `streamer.py` (LSL EEG simulator) | Implemented, untested (optional extension) |
| `detector.py` (band power / focus score) | Planned — stub with TODOs only |
| `agent.py` (contextual bandit) | Planned — stub with TODOs only (optional extension) |

Nothing is **tested** yet. Update this table whenever a component moves to implemented or tested.

**Immediate goal:** a reproducible software proof of concept for personalized EEG channel selection (Level 1). Wait for the developer's approval before substantial implementation.
