"""All-channel baseline on BCI Competition IV 2a, every subject and session (step 2).

Within-session only: each session (T = day 1, E = day 2) is cross-validated on
its own, with the same decoders, window (0.5-4.0 s after the cue) and seeded
5-fold split as baseline.py. Cross-session comes later.
"""

import time

import mne
import numpy as np

from baseline import (
    cross_validate_accuracy,
    log_band_power,
    make_csp_decoder,
    make_decoder,
    task_window,
)
from load_bnci import SESSIONS, prepare_bnci_sessions

SUBJECTS = list(range(1, 10))


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}"


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    start = time.time()
    rows = []
    for subject in SUBJECTS:
        sessions = prepare_bnci_sessions(subject)
        for session, epochs in sessions.items():
            y = epochs.events[:, 2]
            bp = cross_validate_accuracy(log_band_power(epochs), y, decoder=make_decoder())
            csp = cross_validate_accuracy(task_window(epochs), y, decoder=make_csp_decoder())
            rows.append({
                "subject": subject, "session": session, "n_trials": len(y),
                "n_channels": len(epochs.ch_names),
                "majority": max((y == 2).sum(), (y == 3).sum()) / len(y),
                "bp": bp.mean(), "bp_std": bp.std(ddof=1),
                "csp": csp.mean(), "csp_std": csp.std(ddof=1),
            })
            r = rows[-1]
            print(
                f"subject {subject} {session}: {r['n_trials']} trials, {r['n_channels']} ch | "
                f"band power {r['bp']:.3f} +/- {r['bp_std']:.3f} | "
                f"CSP {r['csp']:.3f} +/- {r['csp_std']:.3f}",
                flush=True,
            )

    print("\nWithin-session accuracy, BCI IV 2a left vs right (majority class = 0.500):")
    print("subj   T: band power   CSP  |  E: band power   CSP")
    for subject in SUBJECTS:
        t, e = (next(r for r in rows if r["subject"] == subject and r["session"] == s) for s in SESSIONS)
        print(f"{subject:>4}       {t['bp']:.3f}     {t['csp']:.3f}  |      {e['bp']:.3f}     {e['csp']:.3f}")

    print(f"\nAcross {len(SUBJECTS)} subjects (mean +/- std of per-subject fold means):")
    for session in SESSIONS:
        sel = [r for r in rows if r["session"] == session]
        print(
            f"  session {session}: band power {mean_std([r['bp'] for r in sel])}"
            f"   CSP {mean_std([r['csp'] for r in sel])}"
        )
    print(f"Total time: {time.time() - start:.0f}s")
