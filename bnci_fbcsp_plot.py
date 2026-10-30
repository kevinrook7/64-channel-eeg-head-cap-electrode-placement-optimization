"""Plot the FBCSP vs plain-CSP cross-session result (step 6 figure).

The full FBCSP rerun (bnci_fbcsp.py) takes ~67 min and saves no per-subject
numbers, so this script does NOT recompute anything. It draws the committed mean
accuracies recorded in README.md / AGENTS.md (section 11) for the cross-session
regime (channels chosen + decoders fit on day 1 / T, scored once on all 144
day-2 / E trials, 9 subjects).

Two panels:
  left  -- 4-channel comparison, greedy vs motor strip vs random, plain CSP vs
           FBCSP. This is the headline: at 4 channels greedy's edge over the
           motor strip grows from +0.027 (plain) to +0.069 (FBCSP).
  right -- all 22 channels, plain CSP vs FBCSP (0.741 -> 0.772).

Only the numbers both scripts actually reported are drawn. The 8/16-channel
absolute greedy/motor/random cells were not all recorded for FBCSP (only the
paired differences were), so they are deliberately left off rather than invented.

No leakage and no new computation: this is a presentation of already-committed
cross-session means. Chance is 0.500.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Committed cross-session means (README.md / AGENTS.md section 11), 9 subjects.
# 4-channel absolute cells, greedy / motor / random:
FOUR_CH = {
    "plain CSP": {"greedy": 0.725, "motor": 0.698, "random": 0.663},
    "FBCSP":     {"greedy": 0.761, "motor": 0.692, "random": 0.676},
}
# all-22-channel mean:
FULL_22 = {"plain CSP": 0.741, "FBCSP": 0.772}
METHODS = ["greedy", "motor", "random"]
DECODERS = ["plain CSP", "FBCSP"]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [3, 1]})

    # Left: 4-channel grouped bars (method groups, one bar per decoder).
    x = np.arange(len(METHODS))
    width = 0.38
    for i, dec in enumerate(DECODERS):
        vals = [FOUR_CH[dec][m] for m in METHODS]
        bars = ax1.bar(x + (i - 0.5) * width, vals, width, label=dec)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)
    ax1.axhline(0.5, color="gray", linestyle=":", label="chance (0.5)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["greedy\n(selected)", "motor strip\n(fixed)", "random\n(20 draws)"])
    ax1.set_ylabel("cross-session accuracy (mean across 9 subjects)")
    ax1.set_ylim(0.5, 0.8)
    ax1.set_title("4 channels, day 1 -> day 2")
    ax1.legend(loc="upper right")

    # Right: all-22-channel bars.
    xf = np.arange(len(DECODERS))
    bars = ax2.bar(xf, [FULL_22[d] for d in DECODERS], width=0.6,
                   color=["C0", "C1"])
    for b, d in zip(bars, DECODERS):
        ax2.text(b.get_x() + b.get_width() / 2, FULL_22[d] + 0.004, f"{FULL_22[d]:.3f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.axhline(0.5, color="gray", linestyle=":")
    ax2.set_xticks(xf)
    ax2.set_xticklabels(DECODERS)
    ax2.set_ylim(0.5, 0.8)
    ax2.set_title("all 22 channels")

    fig.suptitle("BCI IV 2a left vs right, cross-session: Filter Bank CSP vs plain CSP")
    fig.tight_layout()
    path = RESULTS_DIR / "bnci_fbcsp.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
