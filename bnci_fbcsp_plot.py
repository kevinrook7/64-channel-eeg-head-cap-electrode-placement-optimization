r"""Plot full FBCSP accuracy-vs-channels curves (step 6 figure).

Reads results/bnci_fbcsp_results.json (written by bnci_fbcsp.py) and draws one
panel per regime -- within-session day 1 (T), within-session day 2 (E), and
cross-session (train T, test E) -- each showing greedy / motor-strip / random
subsets at 4/8/16 channels plus the all-22-channel point, with mean +/- std
error bars across the 9 subjects. Recomputes nothing; it only draws the saved
per-subject numbers.

Run: .\.venv\Scripts\python.exe bnci_fbcsp_plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
JSON_PATH = RESULTS_DIR / "bnci_fbcsp_results.json"

PANELS = [
    ("within_T", "within-session, day 1 (T)"),
    ("within_E", "within-session, day 2 (E)"),
    ("cross_TE", "cross-session, train day 1 -> test day 2"),
]


def panel(ax, rows, sizes, title):
    sizes = np.array(sizes)
    for method, marker in [("greedy", "o"), ("motor", "s"), ("random", "^")]:
        acc = np.array([[r[method][str(k)] for k in sizes] for r in rows])
        ax.errorbar(sizes, acc.mean(0), yerr=acc.std(0, ddof=1), marker=marker,
                    capsize=4, label=method)
    full = np.array([r["full"] for r in rows])
    n_full = np.mean([r["n_channels"] for r in rows])
    ax.errorbar([n_full], [full.mean()], yerr=[full.std(ddof=1)], marker="D",
                color="black", capsize=4, label=f"all channels (~{n_full:.0f})")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance (0.5)")
    ax.set_xticks(list(sizes) + [round(n_full)])
    ax.set_xlabel("number of channels")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.5, 0.95)


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    sizes = data["sizes"]
    results = data["results"]
    n_subj = len(results["cross_TE"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        panel(ax, results[key], sizes, title)
    axes[0].set_ylabel("held-out accuracy (mean ± std across subjects)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"BCI IV 2a left vs right, Filter Bank CSP, {n_subj} subjects")
    fig.tight_layout()
    out = RESULTS_DIR / "bnci_fbcsp.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
