"""Draw Part 2 figures from results/online_results.json -- recomputes nothing.

Two panels:
  left  : accuracy vs recalibration trials (no-recal line, EA line, recal curve),
          averaged across subjects with +/- 1 s.d. band -- the headline figure.
  right : accuracy vs decision latency (all channels vs selected subset),
          averaged across subjects.

Run online/run_online.py first to produce the JSON. Output:
results/online_recalibration.png.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "online_results.json"
OUT = ROOT / "results" / "online_recalibration.png"


def main():
    results = json.loads(RESULTS.read_text())
    n = len(results)

    sizes = results[0]["recalibration"]["sizes"]
    recal = np.array([r["recalibration"]["recal"] for r in results])       # (subjects, sizes)
    no_recal = np.array([r["recalibration"]["no_recal"] for r in results])
    ea = np.array([r["recalibration"]["ea"] for r in results])

    lats = np.array(results[0]["latency"]["latencies_s"])
    acc_all = np.array([r["latency"]["acc_all"] for r in results])
    acc_sel = np.array([r["latency"]["acc_selected"] for r in results])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.6))

    # --- left: accuracy vs recalibration trials ---
    m, s = recal.mean(0), recal.std(0, ddof=1)
    ax1.plot(sizes, m, "o-", color="C0", label="quick recalibration")
    ax1.fill_between(sizes, m - s, m + s, color="C0", alpha=0.15)
    ax1.axhline(no_recal.mean(), ls="--", color="C3", label=f"no recal ({no_recal.mean():.3f})")
    ax1.axhline(ea.mean(), ls=":", color="C2", label=f"causal EA ({ea.mean():.3f})")
    ax1.axhline(0.5, ls="-", color="0.6", lw=0.8, label="chance")
    ax1.set_xlabel("recalibration trials (labeled, from session 2)")
    ax1.set_ylabel("held-out accuracy (session 2)")
    ax1.set_title(f"Quick recalibration of session-1 channels (n={n})")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0.45, 1.0)

    # --- middle: accuracy vs decision latency ---
    for acc, name, c in [(acc_all, "all 22ch", "C1"), (acc_sel, "8 selected", "C0")]:
        mm, ss = acc.mean(0), acc.std(0, ddof=1)
        ax2.plot(lats, mm, "o-", color=c, label=name)
        ax2.fill_between(lats, mm - ss, mm + ss, color=c, alpha=0.15)
    ax2.axhline(0.5, ls="-", color="0.6", lw=0.8, label="chance")
    ax2.set_xlabel("decision latency (s after cue; causal window end)")
    ax2.set_ylabel("cross-session accuracy (session 2)")
    ax2.set_title(f"Accuracy vs decision latency (n={n})")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.45, 1.0)

    # --- right: causal EA running accuracy vs unlabeled trials seen ---
    # Each subject's cumulative-correct / trials-seen, resampled to a common length.
    curves = []
    for r in results:
        seen = np.array(r["recalibration"]["ea_vs_seen"]["trials_seen"], dtype=float)
        cum = np.array(r["recalibration"]["ea_vs_seen"]["cumulative_acc"], dtype=float)
        curves.append(cum / seen)
    L = min(len(c) for c in curves)
    stack = np.stack([c[:L] for c in curves])
    x = np.arange(1, L + 1)
    mm, ss = stack.mean(0), stack.std(0, ddof=1)
    ax3.plot(x, mm, "-", color="C2", label="causal EA (running)")
    ax3.fill_between(x, mm - ss, mm + ss, color="C2", alpha=0.15)
    ax3.axhline(no_recal.mean(), ls="--", color="C3", label=f"no recal ({no_recal.mean():.3f})")
    ax3.axhline(0.5, ls="-", color="0.6", lw=0.8, label="chance")
    ax3.set_xlabel("unlabeled trials seen (causal running reference)")
    ax3.set_ylabel("running accuracy (session 2)")
    ax3.set_title(f"Causal EA vs trials seen (n={n})")
    ax3.legend(fontsize=8)
    ax3.set_ylim(0.45, 1.0)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
