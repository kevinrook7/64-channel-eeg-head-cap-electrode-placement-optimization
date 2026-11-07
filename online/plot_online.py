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
    nr_matched = np.array([r["recalibration"]["no_recal_matched"] for r in results])
    ea_matched = np.array([r["recalibration"]["ea_matched"] for r in results])

    lats = np.array(results[0]["latency"]["latencies_s"])
    acc_all = np.array([r["latency"]["acc_all"] for r in results])
    acc_sel = np.array([r["latency"]["acc_selected"] for r in results])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # 1 --- accuracy vs recalibration trials, ALL on the identical held-out set ---
    # Every line is scored on the same fixed test trials (after the largest k), so
    # the lines are directly comparable. Single deterministic first-k warm-up, so
    # no error bands here (the per-subject spread is panels 2/5).
    m = recal.mean(0)
    ax1.plot(sizes, m, "o-", color="C0", label="quick recalibration (first-k)")
    mc = recal_ea.mean(0)
    ax1.plot(sizes, mc, "s-", color="C4", label="causal EA + recalibration")
    ax1.plot(sizes, nr_matched.mean(0), "^--", color="C3", label="no recal (matched set)")
    ax1.plot(sizes, ea_matched.mean(0), "D:", color="C2", label="causal EA (matched set)")
    ax1.axhline(0.5, ls="-", color="0.6", lw=0.8, label="chance")
    ax1.set_xlabel("recalibration trials (first k of session 2, labeled)")
    ax1.set_ylabel("accuracy on fixed held-out set (session 2)")
    ax1.set_title(f"Recalibration vs EA, identical test trials (n={n})")
    ax1.legend(fontsize=8); ax1.set_ylim(0.6, 0.85)

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

    # 3 --- rolling DIFFERENCE: causal EA minus no-recal, same trials & window.
    #        A rising difference = the EA reference is stabilizing and pulling
    #        ahead; a flat/noisy one = the early cumulative "peak" was trial
    #        difficulty, not reference stabilization (reviewer point 2).
    diff_curves = []
    for r in results:
        rd = r["recalibration"]["ea_vs_seen"].get("rolling_diff")
        if rd is None:
            ra = np.array(r["recalibration"]["ea_vs_seen"]["rolling_acc"], float)
            rb = np.array(r["recalibration"]["ea_vs_seen"].get("rolling_acc_norecal", ra), float)
            rd = ra - rb
        diff_curves.append(np.array(rd, dtype=float))
    L = min(len(c) for c in diff_curves)
    stack = np.stack([c[:L] for c in diff_curves])
    x = np.arange(1, L + 1)
    mm, ss = np.nanmean(stack, 0), np.nanstd(stack, 0, ddof=1)
    roll_w = results[0]["recalibration"]["ea_vs_seen"].get("rolling_window", 20)
    ax3.plot(x, mm, "-", color="C2", label=f"EA − no-recal (rolling {roll_w})")
    ax3.fill_between(x, mm - ss, mm + ss, color="C2", alpha=0.15)
    ax3.axhline(0.0, ls="--", color="C3", label="no difference")
    ax3.set_xlabel(f"unlabeled trials seen (rolling {roll_w}-trial window)")
    ax3.set_ylabel("causal EA − no-recal (rolling)")
    ax3.set_ylim(-0.4, 0.4)
    ax3.set_title(f"Reference stabilization check (n={n})")
    ax3.legend(fontsize=8)

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

    # 6 --- pre-cue control: decode a 0.5 s window of pure rest (must be ~chance) ---
    precue = [r.get("precue_control_acc") for r in results]
    precue = [p for p in precue if p is not None]
    if precue:
        pc = np.array(precue)
        ax6.axhline(0.5, ls="-", color="0.6", lw=0.8, label="chance")
        ax6.scatter(range(len(pc)), pc, color="C5", zorder=3, label="per subject")
        ax6.axhline(pc.mean(), ls="--", color="C5", label=f"mean {pc.mean():.3f}")
        ax6.set_xticks(range(len(pc)))
        ax6.set_xticklabels([r["subject"] for r in results if r.get("precue_control_acc") is not None])
        ax6.set_xlabel("subject")
        ax6.set_ylabel("accuracy, pre-cue window")
        ax6.set_title("Pre-cue control: decode pure rest\n(near chance = no backward leak)", fontsize=10)
        ax6.legend(fontsize=8); ax6.set_ylim(0.3, 0.7)
    else:
        ax6.axis("off")
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
