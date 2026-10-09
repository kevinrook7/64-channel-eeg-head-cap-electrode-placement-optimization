# StayLocked

**Personalized EEG electrode selection for brain-computer interfaces.**

StayLocked asks a simple research question: *can a personalized, task-aware channel-selection optimizer decode intended movements (starting with left- vs right-hand motor imagery) as accurately as a full-channel baseline while using fewer EEG electrodes?*

The first phase is software only: pick channel subsets from public, already-recorded EEG (the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)) and compare them against an all-channel baseline using leakage-safe, subject-specific evaluation. Physical placement recommendations and reconfigurable hardware come later, and only if the software results justify them.

## Status

The channel-selection pipeline hasn't been started yet. Existing scripts from the original blueprint:

- `streamer.py`: a working simulated 4-channel EEG stream over LSL (untested). It's an optional extension for testing real-time infrastructure; simulated data can't show real BCI performance.
- `detector.py`: a stub (TODOs only) for LSL band power and a focus score.
- `agent.py`: a stub (TODOs only) for a contextual-bandit recommender. It's an optional extension.

See the status table in [`AGENTS.md`](AGENTS.md#11-status).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python streamer.py   # broadcasts the "fakeEEG" LSL stream until Ctrl+C
```

## Docs

- [`AGENTS.md`](AGENTS.md): the project vision, research question, pipeline, evaluation rules, and guidance for AI coding agents.
- [`docs/learning-blueprint.md`](docs/learning-blueprint.md): the Build-and-Deconstruct learning approach, reframed for this direction.

## Planned stack

Python, MNE-Python, NumPy/SciPy, scikit-learn. LSL (`pylsl`) only for optional real-time work.
