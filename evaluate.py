"""Compare channel-selection strategies across subjects (Phase 5).

For every subject (within-session, same 5 outer folds as baseline.py):
  - full:   CSP + LR on all eligible channels
  - greedy: channels chosen by greedy forward selection on training trials only
            (nested CV from select_channels.py)
  - motor:  fixed channels around the motor strip (C3/Cz/C4 first), no selection
  - random: random channel subsets of the same size (average of N_RANDOM draws)
Motor and random subsets ignore the labels, so they need no inner loop.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save the figure to a file, no window
import matplotlib.pyplot as plt
import mne
import numpy as np
from joblib import Parallel, delayed  # installed with scikit-learn
from sklearn.model_selection import StratifiedKFold

from baseline import N_SPLITS, SEED, task_window
from fast_csp import trial_covariances
from preprocess import prepare_epochs
from select_channels import SUBSET_SIZES, nested_cv, test_accuracy

SUBJECTS = list(range(1, 11))
N_RANDOM = 20
# Fixed reference: channels ordered outward from the central motor strip.
# The first k channels that survive bad-channel removal form the k-channel set.
MOTOR_ORDER = [
    "C3", "C4", "Cz", "C1", "C2", "C5", "C6", "FC3",               # 8
    "FC4", "CP3", "CP4", "FCz", "CPz", "FC1", "FC2", "CP1",        # 16
    "CP2", "FC5", "FC6", "CP5", "CP6", "T7", "T8", "F3",           # 24
    "F4", "Fz", "F1", "F2", "P3", "P4", "Pz", "P1",                # 32
    "P2", "F5", "F6", "P5", "P6", "FT7", "FT8", "TP7", "TP8",      # spares
]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def motor_subset(names, k):
    return [names.index(ch) for ch in MOTOR_ORDER if ch in names][:k]


def fixed_subset_accuracy(covs, y, subsets, seed=SEED):
    """Mean outer-fold test accuracy for subsets chosen without looking at labels."""
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = [
        test_accuracy(covs[tr], y[tr], covs[te], y[te], chans, seed)
        for tr, te in outer.split(covs, y)
        for chans in subsets
    ]
    return float(np.mean(scores))


def evaluate_subject(subject, covs, y, names, seed=SEED, sizes=SUBSET_SIZES, rng_key=()):
    """Full, greedy, motor-strip and random subsets for one recording.

    `rng_key` makes the random draws differ between recordings of the same
    subject (e.g. the two BCI IV 2a sessions) while staying reproducible.
    """
    start = time.time()
    n_channels = covs.shape[1]
    results, selections, penalized_sizes = nested_cv(covs, y, seed, verbose=False, sizes=sizes)
    rng = np.random.default_rng([seed, subject, *rng_key])

    out = {
        "subject": subject,
        "n_channels": n_channels,
        "majority": max((y == 2).sum(), (y == 3).sum()) / len(y),
        "full": float(np.mean(results["full"])),
        "penalized": float(np.mean(results["penalized"])),
        "penalized_sizes": penalized_sizes,
        "greedy": {k: float(np.mean(results[k])) for k in sizes},
        "motor": {},
        "random": {},
    }
    for k in sizes:
        out["motor"][k] = fixed_subset_accuracy(covs, y, [motor_subset(names, k)], seed)
        draws = [sorted(rng.choice(n_channels, k, replace=False)) for _ in range(N_RANDOM)]
        out["random"][k] = fixed_subset_accuracy(covs, y, draws, seed)

    # Which channels greedy put in the 8-channel set, counted over the 5 outer folds.
    counts = {}
    for selected in selections:
        for idx in selected[:8]:
            counts[names[idx]] = counts.get(names[idx], 0) + 1
    out["greedy8_counts"] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out["seconds"] = time.time() - start
    return out


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}"


def plot_accuracy_vs_channels(rows, path, sizes=SUBSET_SIZES, title=None):
    sizes = np.array(sizes)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, marker in [("greedy", "o"), ("motor", "s"), ("random", "^")]:
        acc = np.array([[r[method][k] for k in sizes] for r in rows])
        ax.errorbar(sizes, acc.mean(0), yerr=acc.std(0, ddof=1), marker=marker,
                    capsize=4, label=method)
    full = np.array([r["full"] for r in rows])
    n_full = np.mean([r["n_channels"] for r in rows])
    ax.errorbar([n_full], [full.mean()], yerr=[full.std(ddof=1)], marker="D",
                color="black", capsize=4, label=f"all channels (~{n_full:.0f})")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance (0.5)")
    ax.set_xticks(list(sizes) + [round(n_full)])
    ax.set_xlabel("number of channels")
    ax.set_ylabel("held-out accuracy (mean ± std across subjects)")
    ax.set_title(title or f"Left vs right imagery, CSP + LR, within-session, {len(rows)} subjects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    start = time.time()

    # Load serially (downloads any missing subjects into data/), then evaluate in parallel.
    data = []
    for subject in SUBJECTS:
        epochs = prepare_epochs(subject)
        y = epochs.events[:, 2]
        data.append((subject, trial_covariances(task_window(epochs)), y, epochs.ch_names))
        print(f"subject {subject}: {len(y)} trials ({(y == 2).sum()} left / "
              f"{(y == 3).sum()} right), {len(epochs.ch_names)} channels")

    rows = Parallel(n_jobs=-1)(delayed(evaluate_subject)(*d) for d in data)

    print("\nHeld-out accuracy per subject (within-session, mean of 5 outer folds):")
    header = "subj  ch  major   full | " + " | ".join(
        f"{m} " + "/".join(str(k) for k in SUBSET_SIZES) for m in ("greedy", "motor", "random")
    )
    print(header)
    for r in rows:
        cells = " | ".join(
            " ".join(f"{r[m][k]:.2f}" for k in SUBSET_SIZES) for m in ("greedy", "motor", "random")
        )
        print(f"{r['subject']:>4} {r['n_channels']:>3}  {r['majority']:.3f}  {r['full']:.3f} | {cells}")

    print(f"\nAcross {len(rows)} subjects (mean +/- std of per-subject accuracy):")
    print(f"  all channels: {mean_std([r['full'] for r in rows])}")
    for k in SUBSET_SIZES:
        print(
            f"  {k:>2} ch  greedy {mean_std([r['greedy'][k] for r in rows])}"
            f"   motor {mean_std([r['motor'][k] for r in rows])}"
            f"   random {mean_std([r['random'][k] for r in rows])}"
        )
    sizes = [s for r in rows for s in r["penalized_sizes"]]
    print(f"  penalized greedy (lambda, {min(sizes)}-{max(sizes)} ch): "
          f"{mean_std([r['penalized'] for r in rows])}")

    print("\nPaired differences (per subject, then mean +/- std):")
    for k in SUBSET_SIZES:
        g = np.array([r["greedy"][k] for r in rows])
        print(
            f"  {k:>2} ch  greedy - full {mean_std(g - [r['full'] for r in rows])}"
            f"   greedy - motor {mean_std(g - [r['motor'][k] for r in rows])}"
            f"   greedy - random {mean_std(g - [r['random'][k] for r in rows])}"
        )

    print("\nGreedy 8-channel picks per subject (channel: folds out of 5, top 8):")
    for r in rows:
        print(f"  {r['subject']:>3}: " + ", ".join(f"{ch} {n}" for ch, n in r["greedy8_counts"][:8]))

    RESULTS_DIR.mkdir(exist_ok=True)
    figure = RESULTS_DIR / "accuracy_vs_channels.png"
    plot_accuracy_vs_channels(rows, figure)
    print(f"\nSaved {figure}")
    print(f"Total time: {time.time() - start:.0f}s")
