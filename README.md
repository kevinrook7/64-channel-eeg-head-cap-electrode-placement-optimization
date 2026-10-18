# StayLocked

**Personalized EEG electrode selection for brain-computer interfaces.**

StayLocked asks a simple research question: *can a personalized, task-aware channel-selection optimizer decode intended movements (starting with left- vs right-hand motor imagery) as accurately as a full-channel baseline while using fewer EEG electrodes?*

The first phase is software only: pick channel subsets from public, already-recorded EEG (the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)) and compare them against an all-channel baseline using leakage-safe, subject-specific evaluation. Physical placement recommendations and reconfigurable hardware come later, and only if the software results justify them.

## Status

Learning-blueprint Phases 1–2 are implemented for subject 1:

- `load_data.py`: `load_subject()` downloads runs 4, 8, and 12, renames T0/T1/T2 per run before concatenating, and returns `raw, events, event_id`.
- `preprocess.py`: attaches a montage, band-pass filters 8–30 Hz, flags and removes bad channels, and cuts left/right epochs. `prepare_epochs()` returns those epochs.
- `baseline.py`: log band power (0.5–4.0 s) and L2 logistic regression, scored with stratified 5-fold cross-validation (seed 0). Subject 1, within-session: accuracy 0.533 ± 0.093 vs 0.511 for always guessing the larger class, so no evidence it beats chance yet.

Next: common spatial patterns (CSP), still in Phase 3. Existing scripts from the original blueprint:

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