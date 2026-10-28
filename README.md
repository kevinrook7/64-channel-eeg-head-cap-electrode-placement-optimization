# MyEEG

**Personalized EEG electrode selection for brain-computer interfaces.**

MyEEG asks a simple research question: *can a personalized, task-aware channel-selection optimizer decode intended movements (starting with left- vs right-hand motor imagery) as accurately as a full-channel baseline while using fewer EEG electrodes?*

The first phase is software only: pick channel subsets from public, already-recorded EEG (the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)) and compare them against an all-channel baseline using leakage-safe, subject-specific evaluation. Physical placement recommendations and reconfigurable hardware come later, and only if the software results justify them.

## Status

Pipeline so far (Phases 1–4 on subject 1, Phase 5 on subjects 1–10):

- `load_data.py`: `load_subject()` downloads runs 4, 8, and 12, renames T0/T1/T2 per run before concatenating, and returns `raw, events, event_id`.
- `preprocess.py`: attaches a montage, band-pass filters 8–30 Hz, flags and removes bad channels, and cuts left/right epochs. `prepare_epochs()` returns those epochs.
- `baseline.py`: log band power (0.5–4.0 s) and L2 logistic regression, scored with stratified 5-fold cross-validation (seed 0). Subject 1, within-session: accuracy 0.556 ± 0.079 vs 0.511 for always guessing the larger class, so no evidence it beats chance yet. Common spatial patterns (CSP, 6 filters fit inside each training fold) + logistic regression on the same folds: 0.800 ± 0.183.

- `select_channels.py`: greedy forward channel selection inside nested cross-validation (channels chosen on training trials only). Subject 1, within-session: 8 channels 0.644 ± 0.183, 16 channels 0.733 ± 0.099, 32 channels 0.644 ± 0.183, vs 0.800 ± 0.183 for all 64. Fewer channels lost accuracy here, and the chosen channels changed from fold to fold.

- `fast_csp.py`: CSP computed from per-trial covariance matrices. Same fold scores as MNE's CSP, and makes greedy selection several times faster.
- `evaluate.py`: compares all channels, greedy subsets, a fixed motor-strip set (C3/C4/Cz outward), and random subsets on subjects 1–10, within-session. All channels 0.589 ± 0.235 across subjects; at 8 channels greedy 0.564, motor strip 0.587, random 0.541. Greedy did not beat the fixed motor strip, and only 3 of 10 subjects decode clearly above chance. Plot: `results/accuracy_vs_channels.png`.

- `load_bnci.py`: loads BCI Competition IV 2a (9 subjects, 2 sessions on different days, 144 left/right trials per session, 22 channels) through MOABB. `prepare_bnci_sessions()` gives both sessions the same channel set for cross-session tests.
- The bad-channel check only drops flat or unusually noisy channels; quiet channels are kept, since they can still carry real signal.

Learning-blueprint Phases 1–5 are now implemented for left vs right imagery. - `bnci_baseline.py`: all-channel baseline on BCI IV 2a, 9 subjects, within-session. CSP 0.773 ± 0.139 on session T and 0.806 ± 0.163 on session E (band power 0.692 / 0.707). 7 of 9 subjects decode well, unlike PhysioNet where only 3 of 10 did.

Next: channel selection (4, 8, 16 channels) on BCI IV 2a, then pick channels on session 1 and test them on session 2. Existing scripts from the original blueprint:

- `streamer.py`: a working simulated 4-channel EEG stream over LSL (untested). It's an optional extension for testing real-time infrastructure; simulated data can't show real BCI performance.
- `detector.py`: a stub (TODOs only) for LSL band power and a focus score.
- `agent.py`: a stub (TODOs only) for a contextual-bandit recommender. It's an optional extension.

See the status table in [`AGENTS.md`](AGENTS.md#11-status).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python streamer.py   # broadcasts the "fakeEEG" LSL stream until Ctrl+C
```

## Docs

- [`AGENTS.md`](AGENTS.md): the project vision, research question, pipeline, evaluation rules, and guidance for AI coding agents.
- [`docs/learning-blueprint.md`](docs/learning-blueprint.md): the Build-and-Deconstruct learning approach, reframed for this direction.

## Planned stack

Python, MNE-Python, NumPy/SciPy, scikit-learn. LSL (`pylsl`) only for optional real-time work.


Notes:

Look into usage of PyTorch
Look into Dr. Kai Keng Ang Filter Bank Common Spatial pattern Algorithm, FBCNet and GANs