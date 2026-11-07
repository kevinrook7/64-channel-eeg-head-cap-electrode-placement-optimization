"""Load BCI Competition IV 2a (MOABB: BNCI2014_001) for left vs right imagery.

9 subjects, 2 sessions on different days ("T" = session 1, "E" = session 2),
6 runs x 48 trials per session, 22 EEG + 3 EOG channels at 250 Hz.

Timing: each trial starts at t = 0, the cue appears at t = 2 s and imagery
lasts until t = 6 s. MOABB's stim channel marks the trial start (its
dataset.interval = [2, 6] is relative to that), so events are shifted to the
cue here. The epochs then line up with the PhysioNet ones (cue at 0 s) and
baseline.task_window() picks 0.5-4.0 s after the cue, i.e. 2.5-6.0 s of the trial.
"""

import os
import warnings
from pathlib import Path

# Keep the download inside the project's (gitignored) data/ folder.
# Set in this process only, so no global MNE config file is changed.
os.environ.setdefault(
    "MNE_DATASETS_BNCI_PATH", str(Path(__file__).resolve().parent / "data")
)

import mne
import numpy as np
from moabb.datasets import BNCI2014_001

from preprocess import epochs_from_raw, filter_and_flag, make_epochs

SESSIONS = {"T": "0train", "E": "1test"}
# MOABB codes -> the codes used for PhysioNet (load_data.EVENT_ID), so the
# rest of the pipeline (labels 2 = left, 3 = right) works unchanged.
LEFT_RIGHT = {1: 2, 2: 3}


def load_session(subject, session):
    """Return (raw, events) for one session; events at cue onset, left=2 / right=3."""
    dataset = BNCI2014_001()
    with warnings.catch_warnings():
        # MOABB uses MNE's deprecated "standard_1005" montage name; harmless here.
        warnings.simplefilter("ignore", FutureWarning)
        runs = dataset.get_data(subjects=[subject])[subject][SESSIONS[session]]
    raw = mne.concatenate_raws([runs[k] for k in sorted(runs, key=int)])

    events = mne.find_events(raw, stim_channel="STI", shortest_event=0, verbose=False)
    events = events[np.isin(events[:, 2], list(LEFT_RIGHT))]
    events[:, 2] = [LEFT_RIGHT[code] for code in events[:, 2]]
    events[:, 0] += int(round(dataset.interval[0] * raw.info["sfreq"]))  # trial start -> cue
    return raw, events


def prepare_bnci_epochs(subject, session="T"):
    """One session on its own: 8-30 Hz, bad-channel check, epochs -0.5 to 4.0 s."""
    raw, events = load_session(subject, session)
    return epochs_from_raw(raw, events)


def prepare_bnci_sessions(subject, causal=False):
    """Both sessions with the SAME channel set, for cross-session use.

    Each session is checked separately; a channel flagged in either session is
    dropped from both, so a model trained on session T can be applied to E.
    The check uses no labels, so looking at session E here leaks nothing.

    `causal` picks the band-pass phase (see preprocess.filter_and_flag). The
    online sweeps pass causal=True so every window is a strictly past-only
    decision; the v1 offline baselines keep the default zero-phase filter.
    """
    filtered, events, bads = {}, {}, set()
    for session in SESSIONS:
        raw, events[session] = load_session(subject, session)
        filtered[session], session_bads = filter_and_flag(raw, causal=causal)
        bads.update(session_bads)
    epochs = {s: make_epochs(filtered[s], events[s], sorted(bads)) for s in SESSIONS}
    assert epochs["T"].ch_names == epochs["E"].ch_names
    return epochs


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    sessions = prepare_bnci_sessions(1)
    for session, epochs in sessions.items():
        y = epochs.events[:, 2]
        print(
            f"subject 1, session {session}: {len(y)} trials "
            f"({(y == 2).sum()} left / {(y == 3).sum()} right), "
            f"{len(epochs.ch_names)} EEG channels, {epochs.info['sfreq']:.0f} Hz, "
            f"shape {epochs.get_data().shape}"
        )
    print("channels:", ", ".join(epochs.ch_names))
