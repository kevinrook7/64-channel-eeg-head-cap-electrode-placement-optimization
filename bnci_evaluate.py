"""Channel selection on BCI Competition IV 2a, within-session (step 3).

Same comparison as evaluate.py (all channels vs greedy vs motor strip vs random,
greedy chosen by nested CV on training trials only), but with subset sizes
4 / 8 / 16 because the montage only has 22 channels. Each session (T = day 1,
E = day 2) is evaluated on its own; cross-session is step 4.
"""

import time

import mne
import numpy as np
from joblib import Parallel, delayed  # installed with scikit-learn

from baseline import task_window
from evaluate import RESULTS_DIR, evaluate_subject, mean_std, motor_subset, plot_accuracy_vs_channels
from fast_csp import trial_covariances
from load_bnci import SESSIONS, prepare_bnci_sessions

SUBJECTS = list(range(1, 10))
BNCI_SIZES = [4, 8, 16]
METHODS = ("greedy", "motor", "random")


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    start = time.time()

    jobs = []
    for subject in SUBJECTS:
        for session, epochs in prepare_bnci_sessions(subject).items():
            jobs.append((subject, session, trial_covariances(task_window(epochs)),
                         epochs.events[:, 2], epochs.ch_names))
    names = jobs[0][4]
    for k in BNCI_SIZES:
        print(f"motor strip {k}ch: {', '.join(names[i] for i in motor_subset(names, k))}")

    outs = Parallel(n_jobs=-1)(
        delayed(evaluate_subject)(subject, covs, y, ch, sizes=BNCI_SIZES, rng_key=(ord(session),))
        for subject, session, covs, y, ch in jobs
    )
    for (subject, session, *_), out in zip(jobs, outs):
        out["session"] = session

    sizes_txt = "/".join(str(k) for k in BNCI_SIZES)
    for session in SESSIONS:
        rows = [r for r in outs if r["session"] == session]
        print(f"\nSession {session}, within-session held-out accuracy (mean of 5 outer folds, chance 0.500):")
        print("subj  full | " + " | ".join(f"{m} {sizes_txt}" for m in METHODS))
        for r in rows:
            cells = " | ".join(" ".join(f"{r[m][k]:.2f}" for k in BNCI_SIZES) for m in METHODS)
            print(f"{r['subject']:>4}  {r['full']:.3f} | {cells}")

        print(f"Across {len(rows)} subjects (mean +/- std):")
        print(f"  all {rows[0]['n_channels']} channels: {mean_std([r['full'] for r in rows])}")
        for k in BNCI_SIZES:
            print(f"  {k:>2} ch  " + "   ".join(f"{m} {mean_std([r[m][k] for r in rows])}" for m in METHODS))
        pen = [s for r in rows for s in r["penalized_sizes"]]
        print(f"  penalized greedy ({min(pen)}-{max(pen)} ch): {mean_std([r['penalized'] for r in rows])}")

        print("Paired differences (per subject, then mean +/- std):")
        full = np.array([r["full"] for r in rows])
        for k in BNCI_SIZES:
            g = np.array([r["greedy"][k] for r in rows])
            m = np.array([r["motor"][k] for r in rows])
            rnd = np.array([r["random"][k] for r in rows])
            print(f"  {k:>2} ch  greedy - full {mean_std(g - full)}   greedy - motor {mean_std(g - m)}"
                  f"   greedy - random {mean_std(g - rnd)}   motor - full {mean_std(m - full)}")

        print("Greedy 8-channel picks (channel: folds out of 5, top 8):")
        for r in rows:
            print(f"  {r['subject']:>3}: " + ", ".join(f"{ch} {n}" for ch, n in r["greedy8_counts"][:8]))

        RESULTS_DIR.mkdir(exist_ok=True)
        figure = RESULTS_DIR / f"bnci_accuracy_vs_channels_{session}.png"
        plot_accuracy_vs_channels(
            rows, figure, sizes=BNCI_SIZES,
            title=f"BCI IV 2a left vs right, CSP + LR, within-session {session}, {len(rows)} subjects",
        )
        print(f"Saved {figure}")

    print(f"\nTotal time: {time.time() - start:.0f}s")
