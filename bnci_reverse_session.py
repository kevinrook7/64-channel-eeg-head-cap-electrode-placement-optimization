"""Reverse cross-session channel selection on BCI IV 2a (step 4b).

Step 4 (bnci_cross_session.py) trained on day 1 (T) and tested on day 2 (E).
This runs the SAME procedure the other way: pick channels and fit every decoder
on day 2 (E), then score once on all 144 day-1 (T) trials. Doing both directions
doubles the cross-session evidence -- if greedy beats the motor strip and random
in T->E but not in E->T, the step-4 win was likely one lucky split.

It reuses cross_session_subject() unchanged by passing the sessions swapped:
cross_session_subject(subject, c_e, y_e, c_t, y_t, ...) makes E the training
session ("picks_t"/day-1 slots now hold the E picks) and T the test session.

NO LEAKAGE: channels are chosen and decoders fit on the training session (E here)
only; the test session (T) is scored once. Same guarantee as step 4.
"""

import time

import mne
import numpy as np
from joblib import Parallel, delayed  # installed with scikit-learn

from baseline import task_window
from bnci_cross_session import cross_session_subject
from bnci_evaluate import BNCI_SIZES, METHODS, SUBJECTS
from evaluate import mean_std
from fast_csp import trial_covariances
from load_bnci import prepare_bnci_sessions
from select_channels import LAMBDA


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    start = time.time()

    jobs = []
    for subject in SUBJECTS:
        sessions = prepare_bnci_sessions(subject)   # same channel set in T and E
        t, e = sessions["T"], sessions["E"]
        # Swapped: E is the training session, T is the held-out test session.
        jobs.append((subject, trial_covariances(task_window(e)), e.events[:, 2],
                     trial_covariances(task_window(t)), t.events[:, 2], e.ch_names))

    rows = Parallel(n_jobs=-1)(delayed(cross_session_subject)(*job) for job in jobs)

    sizes_txt = "/".join(str(k) for k in BNCI_SIZES)
    print("\nReverse cross-session: trained on day 2 (E), tested once on all 144 day-1 (T) trials")
    print("(chance 0.500; with 144 test trials one subject's score has a standard error of ~0.03-0.04)")
    print("subj  full | " + " | ".join(f"{m} {sizes_txt}" for m in METHODS) + " | penalized")
    for r in rows:
        cells = " | ".join(" ".join(f"{r[m][k]:.2f}" for k in BNCI_SIZES) for m in METHODS)
        print(f"{r['subject']:>4}  {r['full']:.3f} | {cells} | {r['penalized']:.2f} ({r['penalized_size']}ch)")

    print(f"\nAcross {len(rows)} subjects (mean +/- std):")
    print(f"  all {rows[0]['n_channels']} channels: {mean_std([r['full'] for r in rows])}")
    for k in BNCI_SIZES:
        print(f"  {k:>2} ch  " + "   ".join(f"{m} {mean_std([r[m][k] for r in rows])}" for m in METHODS))
    print(f"  penalized greedy (lambda={LAMBDA}, sizes {[r['penalized_size'] for r in rows]}): "
          f"{mean_std([r['penalized'] for r in rows])}")

    print("\nPaired differences (per subject, then mean +/- std; 'wins' = subjects where greedy is higher):")
    full = np.array([r["full"] for r in rows])
    for k in BNCI_SIZES:
        g, m, rnd = (np.array([r[x][k] for r in rows]) for x in METHODS)
        print(f"  {k:>2} ch  greedy - full {mean_std(g - full)}"
              f"   greedy - motor {mean_std(g - m)} (wins {int((g > m).sum())}, ties {int((g == m).sum())})"
              f"   greedy - random {mean_std(g - rnd)} (wins {int((g > rnd).sum())})")

    print(f"\nTotal time: {time.time() - start:.0f}s")
