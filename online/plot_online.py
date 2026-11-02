"""Draw Part 2 figures from results/online_results.json -- recomputes nothing.

Five panels (2x3 grid, last cell blank):
  1 recalibration trials: quick recal, EA+recal combo, no-recal and causal-EA lines
  2 decision latency: all 22ch vs 8 selected
  3 causal EA running accuracy vs unlabeled trials seen
  4 window-length sweep: accuracy vs latency for several window lengths (pre-cue
    windows dropped), which separates "more data" from "pre-cue contamination"
  5 paired dot plot: causal EA minus no-recal per subject, with the Wilcoxon p

The JSON is a list of per-subject records, optionally followed by one trailing
{"paired_stats_ea_vs_norecal": ...} record; subject records are the ones with a
"subject" key. Run online/run_online.py first. Output: results/online_recalibration.png.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "online_results.json"
OUT = ROOT / "results" / "online_recalibration.png"


def main():
    raw = json.loads(RESULTS.read_text())
    results = [r for r in raw if "subject" in r]                 # drop trailing stats record
    stats = next((r["paired_stats_ea_vs_norecal"] for r in raw
                  if "paired_stats_ea_vs_norecal" in r), None)
    n = len(results)

    sizes = results[0]["recalibration"]["sizes"]
    recal = np.array([r["recalibration"]["recal"] for r in results])
    recal_ea = np.array([r["recalibration"]["recal_ea"] for r in results])
    no_recal = np.array([r["recalibration"]["no_recal"] for r in results])
    ea = np.array([r["recalibration"]["ea"] for r in results])

    lats = np.array(results[0]["latency"]["latencies_s"])
    acc_all = np.array([r["latency"]["acc_all"] for r in results])
    acc_sel = np.array([r["latency"]["acc_selected"] for r in results])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # 1 --- accuracy vs recalibration trials (plain + EA+recal combo) ---
    m, s = recal.mean(0), recal.std(0, ddof=1)
    ax1.plot(sizes, m, "o-", color="C0", label="quick recalibration")
    ax1.fill_between(sizes, m - s, m + s, color="C0", alpha=0.15)
    mc, sc = recal_ea.mean(0), recal_ea.std(0, ddof=1)
    ax1.plot(sizes, mc, "s-", color="C4", label="causal EA + recalibration")
    ax1.fill_between(sizes, mc - sc, mc + sc, color="C4", alpha=0.12)
    ax1.axhline(no_recal.mean(), ls="--", color="C3", label=f"no recal ({no_recal.mean():.3f})")
    ax1.axhline(ea.mean(), ls=":", color="C2", label=f"causal EA ({ea.mean():.3f})")
    ax1.axhline(0.5, ls="-", color="0.6", lw=0.8, label="chance")
    ax1.set_xlabel("recalibration trials (labeled, from session 2)")
    ax1.set_ylabel("held-out accuracy (session 2)")
    ax1.set_title(f"Recalibration & EA+recal combo (n={n})")
    ax1.legend(fontsize=8); ax1.set_ylim(0.45, 1.0)

    # 2 --- accuracy vs decision latency ---
    for acc, name, c in [(acc_all, "all 22ch", "C1"), (acc_sel, "8 selected", "C0")]:
        mm, ss = acc.mean(0), acc.std(0, ddof=1)
        ax2.plot(lats, mm, "o-", color=c, label=name)
        ax2.fill_between(lats, mm - ss, mm + ss, color=c, alpha=0.15)
    ax2.axhline(0.5, ls="-", color="0.6", lw=0.8, label="chance")
    ax2.set_xlabel("decision latency (s after cue; 2.0 s window)")
    ax2.set_ylabel("cross-session accuracy (session 2)")
    ax2.set_title(f"Accuracy vs decision latency (n={n})")
    ax2.legend(fontsize=8); ax2.set_ylim(0.45, 1.0)

    # 3 --- causal EA running accuracy vs unlabeled trials seen ---
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
    ax3.axvline(10, ls=":", color="0.5", lw=0.8, label="reference stable (~10)")
    ax3.axhline(0.5, ls="-", color="0.6", lw=0.8, label="chance")
    ax3.set_xlabel("unlabeled trials seen (causal running reference)")
    ax3.set_ylabel("running accuracy (session 2)")
    ax3.set_title(f"Causal EA vs trials seen (n={n})")
    ax3.legend(fontsize=8); ax3.set_ylim(0.45, 1.0)

    # 4 --- window-length sweep ---
    win_keys = sorted(results[0]["winlen_sweep"].keys(), key=float)
    for wl in win_keys:
        # average each window length's curve over subjects on its shared latencies
        all_lat = [np.array(r["winlen_sweep"][wl][0]) for r in results]
        all_acc = [np.array(r["winlen_sweep"][wl][1]) for r in results]
        common = sorted(set(all_lat[0]).intersection(*[set(a.tolist()) for a in all_lat[1:]]))
        if not common:
            continue
        means = []
        for lat in common:
            vals = [acc[np.where(l == lat)[0][0]] for l, acc in zip(all_lat, all_acc)]
            means.append(np.mean(vals))
        ax4.plot(common, means, "o-", label=f"{wl}s window")
    ax4.axhline(0.5, ls="-", color="0.6", lw=0.8, label="chance")
    ax4.set_xlabel("decision latency (s after cue; pre-cue windows dropped)")
    ax4.set_ylabel("accuracy (8 selected ch)")
    ax4.set_title(f"Window-length sweep (n={n})")
    ax4.legend(fontsize=8); ax4.set_ylim(0.45, 1.0)

    # 5 --- paired dot plot: causal EA minus no-recal per subject ---
    diffs = ea - no_recal
    order = np.argsort(diffs)
    ax5.axhline(0, ls="-", color="0.6", lw=0.8)
    ax5.scatter(range(n), diffs[order], color=["C2" if d > 0 else "C3" for d in diffs[order]], zorder=3)
    ax5.set_xticks(range(n))
    ax5.set_xticklabels([results[i]["subject"] for i in order])
    ax5.set_xlabel("subject (sorted by effect)")
    ax5.set_ylabel("causal EA − no-recal")
    title = f"Paired effect per subject (n={n})"
    if stats and stats.get("wilcoxon_p") is not None:
        title += f"\nmean {stats['mean_diff']:+.3f}, {stats['n_improve']}/{stats['n']} up, Wilcoxon p={stats['wilcoxon_p']:.3f}"
    ax5.set_title(title, fontsize=10)

    ax6.axis("off")
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
