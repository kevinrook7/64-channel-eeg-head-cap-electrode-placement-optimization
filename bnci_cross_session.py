"""Cross-session channel selection on BCI Competition IV 2a (step 4).

Train on day 1 (session T), test on day 2 (session E), as a real BCI would be used:
  - greedy channels are chosen with inner 5-fold CV on session T only
  - every decoder (all channels, greedy, motor strip, random) is fit on all of T
  - accuracy is measured once on all 144 trials of E, which nothing has seen

Stability check: greedy is also run on session E alone, purely to compare which
channels each day picks. Those E picks are never used to score anything.
"""

import time

import mne
import numpy as np
from joblib import Parallel, delayed  # installed with scikit-learn

from baseline import SEED, task_window
from bnci_evaluate import BNCI_SIZES, METHODS, SUBJECTS
from evaluate import N_RANDOM, RESULTS_DIR, mean_std, motor_subset, plot_accuracy_vs_channels
from fast_csp import trial_covariances
from load_bnci import prepare_bnci_sessions
from select_channels import LAMBDA, greedy_forward, penalized_size, test_accuracy


def cross_session_subject(subject, c_t, y_t, c_e, y_e, names, seed=SEED):
    start = time.time()
    n_channels = c_t.shape[1]
    max_size = max(BNCI_SIZES)

    def day2_accuracy(channels):
        return test_accuracy(c_t, y_t, c_e, y_e, list(channels), seed)

    picks_t, path = greedy_forward(c_t, y_t, max_size, seed)   # day 1 only
    k_pen = penalized_size(path)
    picks_e, _ = greedy_forward(c_e, y_e, max_size, seed)      # stability check only
    rng = np.random.default_rng([seed, subject, 2])

    out = {
        "subject": subject,
        "n_channels": n_channels,
        "full": day2_accuracy(range(n_channels)),
        "penalized": day2_accuracy(picks_t[:k_pen]),
        "penalized_size": k_pen,
        "greedy": {}, "motor": {}, "random": {}, "overlap": {},
        "picks_t": [names[i] for i in picks_t[:8]],
        "picks_e": [names[i] for i in picks_e[:8]],
    }
    for k in BNCI_SIZES:
        out["greedy"][k] = day2_accuracy(picks_t[:k])
        out["motor"][k] = day2_accuracy(motor_subset(names, k))
        draws = [sorted(rng.choice(n_channels, k, replace=False)) for _ in range(N_RANDOM)]
        out["random"][k] = float(np.mean([day2_accuracy(d) for d in draws]))
        out["overlap"][k] = len(set(picks_t[:k]) & set(picks_e[:k]))
    out["seconds"] = time.time() - start
    return out


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    start = time.time()

    jobs = []
    for subject in SUBJECTS:
        sessions = prepare_bnci_sessions(subject)   # same channel set in T and E
        t, e = sessions["T"], sessions["E"]
        jobs.append((subject, trial_covariances(task_window(t)), t.events[:, 2],
                     trial_covariances(task_window(e)), e.events[:, 2], t.ch_names))

    rows = Parallel(n_jobs=-1)(delayed(cross_session_subject)(*job) for job in jobs)

    sizes_txt = "/".join(str(k) for k in BNCI_SIZES)
    print("\nCross-session accuracy: trained on day 1 (T), tested once on all 144 day-2 (E) trials")
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

    n = rows[0]["n_channels"]
    print("\nDo day 1 and day 2 pick the same channels? (greedy run on each day separately)")
    print("  overlap of the k-channel sets; chance level for two random sets is k^2/" + str(n))
    for k in BNCI_SIZES:
        ov = [r["overlap"][k] for r in rows]
        print(f"  {k:>2} ch: mean overlap {np.mean(ov):.1f} of {k} (chance {k * k / n:.1f}), per subject {ov}")
    print("  first 8 picks, day 1 | day 2:")
    for r in rows:
        print(f"  {r['subject']:>3}: {', '.join(r['picks_t'])} | {', '.join(r['picks_e'])}")

    RESULTS_DIR.mkdir(exist_ok=True)
    figure = RESULTS_DIR / "bnci_cross_session.png"
    plot_accuracy_vs_channels(
        rows, figure, sizes=BNCI_SIZES,
        title=f"BCI IV 2a left vs right, CSP + LR, cross-session T -> E, {len(rows)} subjects",
    )
    print(f"\nSaved {figure}")
    print(f"Total time: {time.time() - start:.0f}s")
