"""Print the README's Part 2 numbers straight from results/online_results.json.

Single source of truth: paste from this, never transcribe by hand. Run after
online/run_online.py. Prints the results table, the per-subject paired stats, and
the non-causal-vs-causal comparison (reads docs/online_results_noncausal.json if
present).

    python -m online.print_readme_numbers
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAUSAL = ROOT / "results" / "online_results.json"
NONCAUSAL = ROOT / "docs" / "online_results_noncausal.json"


def _load(path):
    raw = json.loads(Path(path).read_text())
    subs = [r for r in raw if "subject" in r]
    stats = next((r["paired_stats_ea_vs_norecal"] for r in raw
                  if "paired_stats_ea_vs_norecal" in r), None)
    return subs, stats


def _mean(subs, *keys):
    vals = []
    for r in subs:
        d = r
        for k in keys:
            d = d[k]
        vals.append(d)
    return float(np.mean(vals))


def main():
    subs, stats = _load(CAUSAL)
    n = len(subs)
    rc0 = subs[0]["recalibration"]
    sizes = rc0["sizes"]
    scored = rc0.get("scored_trials", "?")
    scored_from = rc0.get("scored_from_trial", "?")

    def col(key):
        return np.array([r["recalibration"][key] for r in subs]).mean(0)

    print(f"# Part 2 results (n={n}, causal filter), scored on fixed held-out set")
    print(f"# = trials {scored_from}..end ({scored} trials), identical across all conditions\n")
    print(f"{'condition':42s} mean acc (n={n})")
    print("-" * 60)
    nr_m, ea_m, rec, rec_ea = col("no_recal_matched"), col("ea_matched"), col("recal"), col("recal_ea")
    print(f"{'no recalibration':42s} {nr_m[0]:.3f}")
    print(f"{'causal EA (running ref, 0 labels)':42s} {ea_m[0]:.3f}")
    print(f"\n# full stream (all trials; latency analyses):")
    print(f"{'no recalibration':42s} {_mean(subs,'recalibration','no_recal'):.3f}")
    print(f"{'causal EA (running ref, 0 labels)':42s} {_mean(subs,'recalibration','ea'):.3f}")
    print(f"{'EA transductive (contrast only)':42s} {_mean(subs,'recalibration','ea_transductive'):.3f}")
    print(f"\n# quick recalibration on the fixed held-out set:")
    for i, k in enumerate(sizes):
        print(f"{'  quick recalibration, k=%-3d' % k:42s} {rec[i]:.3f}   "
              f"(no-recal {nr_m[i]:.3f} | EA {ea_m[i]:.3f} | EA+recal {rec_ea[i]:.3f}, same trials)")
    print(f"\ndecode latency: {_mean(subs,'decode_ms_per_window_end_to_end'):.3f} ms/window end-to-end")
    print(f"pre-cue control: {_mean(subs,'recalibration','no_recal') and np.mean([r['precue_control_acc'] for r in subs]):.3f} (chance = no leak)")

    # Paired test on the FIXED held-out set (the same trials as the table above).
    # The matched columns are identical at every k (same trials, no labels), so k index 0.
    from scipy.stats import wilcoxon
    ea_s = np.array([r["recalibration"]["ea_matched"] for r in subs])
    nr_s = np.array([r["recalibration"]["no_recal_matched"] for r in subs])
    assert np.allclose(ea_s, ea_s[:, :1]) and np.allclose(nr_s, nr_s[:, :1]), \
        "matched EA/no-recal differ across k -- check scoring"
    md = np.round(ea_s[:, 0] - nr_s[:, 0], 10)
    mup, mtie, mdown = int((md > 0).sum()), int((md == 0).sum()), int((md < 0).sum())
    mp = wilcoxon(md).pvalue  # zero_method='wilcox' drops ties
    print(f"\npaired causal EA − no-recal, fixed held-out set: {mup} up, {mtie} tie, {mdown} down; "
          f"mean {md.mean():+.3f}, Wilcoxon on {len(md) - mtie} non-tied, p={mp:.3f}")
    print("  per subject: " + ", ".join(f"S{r['subject']} {d:+.3f}" for r, d in zip(subs, md)))

    if stats:
        diffs = np.array(stats["per_subject_diff"])
        up = int((diffs > 0).sum()); tie = int((diffs == 0).sum()); down = int((diffs < 0).sum())
        print(f"paired causal EA − no-recal, full stream: {up} up, {tie} tie, {down} down; "
              f"mean {stats['mean_diff']:+.3f}, Wilcoxon on {stats['n'] - tie} non-tied, "
              f"p={stats['wilcoxon_p']:.3f}")

    # latency extremes for the non-causal-vs-causal table
    lats = subs[0]["latency"]["latencies_s"]
    acc = np.array([r["latency"]["acc_selected"] for r in subs]).mean(0)
    print(f"latency: earliest {lats[0]:.1f}s = {acc[0]:.3f}, latest {lats[-1]:.1f}s = {acc[-1]:.3f}")

    if NONCAUSAL.exists():
        nsubs, nstats = _load(NONCAUSAL)
        print("\n# Non-causal (zero-phase) for comparison:")
        print(f"no-recal {_mean(nsubs,'recalibration','no_recal'):.3f} | "
              f"EA {_mean(nsubs,'recalibration','ea'):.3f}")
        nlats = nsubs[0]["latency"]["latencies_s"]
        nacc = np.array([r["latency"]["acc_selected"] for r in nsubs]).mean(0)
        print(f"latency earliest {nlats[0]:.1f}s = {nacc[0]:.3f}, latest {nlats[-1]:.1f}s = {nacc[-1]:.3f}")
        if nstats:
            nd = np.array(nstats["per_subject_diff"])
            print(f"paired: {int((nd>0).sum())} up / {int((nd==0).sum())} tie / {int((nd<0).sum())} down, "
                  f"mean {nstats['mean_diff']:+.3f}, p={nstats['wilcoxon_p']:.3f}")


if __name__ == "__main__":
    main()
