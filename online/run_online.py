"""Run Part 2 over all 9 BCI IV 2a subjects and save the exact numbers the README
quotes to results/online_results.json.

Two sweeps, both causal and leakage-safe:

  A. Latency sweep  -- accuracy vs decision latency (time since cue), all 22
     channels and the session-1-selected subset. Answers "how soon can it decide?"
  B. Recalibration sweep -- accuracy vs number of labeled recalibration trials
     (0/5/10/20/40), with no-recal / EA / recal conditions, for the selected
     subset. Answers "can a small session-1 channel set recover on session 2
     with a short warm-up?"

Channel selection (step 1 of the honest design) runs on SESSION 1 ONLY, reusing
v1's greedy forward selection (select_channels.greedy_forward) on session-1
imagery-window covariances. Session 2 never informs the choice.

Two classes only (left/right), as load_bnci restricts. This is NOT a quick run:
loading 9 subjects through MOABB + MNE needs memory, and the sweeps refit CSP
many times. Launch it in the background with a log when memory is free, or on a
machine with headroom -- never inline on a starved host. It writes JSON only; the
figure is drawn separately by online/plot_online.py (recomputes nothing).
"""

import json
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results" / "online_results.json"

# Causal window used for the recalibration sweep and the online demo.
WIN_LEN = 2.0          # seconds of look-back
DECISION_LATENCY = 2.0  # window ends 2.0 s after cue (so it spans cue+0.0 .. cue+2.0 s)
CUE_OFFSET = 0.5        # BNCI epochs start 0.5 s before the cue (tmin = -0.5)
LATENCIES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # decision latencies for sweep A
WIN_LENS = [0.5, 1.0, 1.5, 2.0]  # window lengths for the window-length sweep (A2)
N_SELECT = 8           # channels to select on session 1 (v1's practical subset)
SEED = 0
CAUSAL = True          # online sweeps use a causal (phase='minimum') band-pass so
                       # every windowed decision truly uses only past samples.


def select_on_session1(covs_tr, y_tr, n_select, seed=SEED):
    """Greedy forward selection on session 1 only (no test-session leakage)."""
    from select_channels import greedy_forward
    selected, _ = greedy_forward(covs_tr, y_tr, n_select, seed)
    return selected


def run_subject(subject):
    """All sweeps for one subject. Returns a JSON-able dict."""
    import mne
    from baseline import task_window
    from fast_csp import trial_covariances
    from load_bnci import prepare_bnci_sessions
    from online.causal_decoder import latency_curve
    from online.recalibration import recalibration_curve, window_covs_for

    mne.set_log_level("WARNING")
    sess = prepare_bnci_sessions(subject, causal=CAUSAL)
    sfreq = sess["T"].info["sfreq"]
    X_tr, y_tr = task_window(sess["T"]), sess["T"].events[:, 2]
    X_te, y_te = task_window(sess["E"]), sess["E"].events[:, 2]
    n_ch = X_tr.shape[1]

    # Channel selection: session 1 only, on the full imagery window covariances.
    covs_full_tr = trial_covariances(X_tr)
    selected = select_on_session1(covs_full_tr, y_tr, N_SELECT)

    # --- Sweep A: accuracy vs latency (all channels and selected subset) ---
    lat_all = latency_curve(X_tr, y_tr, X_te, y_te, sfreq, LATENCIES,
                            win_len=WIN_LEN, cue_offset=CUE_OFFSET, channels=None, seed=SEED)
    lat_sel = latency_curve(X_tr, y_tr, X_te, y_te, sfreq, LATENCIES,
                            win_len=WIN_LEN, cue_offset=CUE_OFFSET, channels=selected, seed=SEED)

    # --- Sweep A2: window-LENGTH sweep (selected subset), pre-cue windows dropped ---
    from online.causal_decoder import window_length_sweep
    winlen_sweep = window_length_sweep(X_tr, y_tr, X_te, y_te, sfreq,
                                       WIN_LENS, LATENCIES, cue_offset=CUE_OFFSET,
                                       channels=selected, seed=SEED, drop_precue=True)

    # --- Sweep B: accuracy vs recalibration trials (selected subset) ---
    covs_tr = window_covs_for(X_tr, sfreq, WIN_LEN, DECISION_LATENCY, CUE_OFFSET)
    covs_te = window_covs_for(X_te, sfreq, WIN_LEN, DECISION_LATENCY, CUE_OFFSET)

    # Measure DECODE latency: wall-clock to score one held-out window of covariances.
    from online.recalibration import fit_decoder, score
    # Measure END-TO-END decode latency per window: take the raw causal window
    # samples and time covariance + channel-subset + classify (what a live device
    # runs each step). Band-pass filtering is done once per session up front in
    # this offline pipeline, not per window, so it is excluded and that is stated.
    from online.causal_decoder import window_bounds
    dec = fit_decoder(covs_tr, y_tr, selected, SEED)
    bnd = window_bounds(sfreq, WIN_LEN, [DECISION_LATENCY], CUE_OFFSET)[0]
    start, stop = bnd
    one_window_raw = X_te[:, :, start:stop]            # raw samples per trial for this window
    sel = np.asarray(selected)
    t0 = time.perf_counter()
    for i in range(len(y_te)):
        w = one_window_raw[i]                           # (channels, times)
        cov = (w @ w.T) / w.shape[1]                    # covariance from raw samples
        sub = cov[sel[:, None], sel[None, :]][None]     # channel subset
        _ = dec.predict(sub)                            # CSP log-power + LR
    decode_ms = (time.perf_counter() - t0) / len(y_te) * 1000  # end-to-end, ms/window

    recal = recalibration_curve(covs_tr, y_tr, covs_te, y_te, channels=selected, seed=SEED)

    # --- Pre-cue control (reviewer point 2): decode a 0.5 s window that ends at
    # the cue, i.e. pure pre-cue rest. Uses the FULL epoch arrays (task_window
    # crops to post-cue, so it can't reach rest). Must come out ~0.5 if nothing
    # leaks backward; a clearly-above-chance number means contamination.
    from online.causal_decoder import window_covariances
    Xf_tr, Xf_te = sess["T"].get_data(), sess["E"].get_data()  # epoch starts tmin=-0.5
    pre_bounds = window_bounds(sfreq, 0.5, [0.0], CUE_OFFSET)  # 0.5 s window ending AT cue
    pctr = window_covariances(Xf_tr, pre_bounds)
    pcte = window_covariances(Xf_te, pre_bounds)
    precue_acc = None
    if 0 in pctr and 0 in pcte:
        from online.causal_decoder import fit_window_decoder, score_window
        wdec = fit_window_decoder(pctr[0], y_tr, selected, SEED)
        precue_acc = float(score_window(wdec, pcte[0], y_te, selected))

    return {
        "subject": subject,
        "n_channels": int(n_ch),
        "selected_channels": [sess["T"].ch_names[i] for i in selected],
        "causal_filter": CAUSAL,
        "precue_control_acc": precue_acc,
        "latency": {
            "latencies_s": lat_all[0].tolist(),
            "acc_all": lat_all[1].tolist(),
            "acc_selected": lat_sel[1].tolist(),
        },
        "winlen_sweep": {str(wl): winlen_sweep[wl] for wl in winlen_sweep},
        "recalibration": recal,
        "decode_ms_per_window_end_to_end": round(decode_ms, 3),
        "decode_ms_covers": "covariance + channel-subset + CSP log-power + LR; excludes one-off per-session band-pass",
        "window_len_s": WIN_LEN,
        "decision_latency_s": DECISION_LATENCY,
    }


def main(subjects=range(1, 10)):
    results = []
    for s in subjects:
        start = time.time()
        r = run_subject(s)
        results.append(r)
        rc = r["recalibration"]
        print(
            f"subject {s}: no-recal {rc['no_recal']:.3f} | EA {rc['ea']:.3f} | "
            f"recal@20 {rc['recal'][rc['sizes'].index(20)]:.3f} | "
            f"decode {r['decode_ms_per_window_end_to_end']:.3f} ms/window ({time.time()-start:.0f}s)"
        )
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))

    # Cross-subject means for the README.
    def mean_over(key_path):
        vals = []
        for r in results:
            d = r
            for k in key_path:
                d = d[k]
            vals.append(d)
        return float(np.mean(vals))

    print("\n--- cross-subject means (n=%d) ---" % len(results))
    sizes = results[0]["recalibration"]["sizes"]
    print("no-recal:", round(mean_over(["recalibration", "no_recal"]), 3))
    print("EA:      ", round(mean_over(["recalibration", "ea"]), 3))
    for i, k in enumerate(sizes):
        m = float(np.mean([r["recalibration"]["recal"][i] for r in results]))
        print(f"recal@{k:<3}:", round(m, 3))
    print("decode ms/window (end-to-end):", round(mean_over(["decode_ms_per_window_end_to_end"]), 3))

    # --- paired statistics: causal EA minus no-recal, per subject ---
    from scipy.stats import wilcoxon
    diffs = np.array([r["recalibration"]["ea"] - r["recalibration"]["no_recal"] for r in results])
    n_up = int((diffs > 0).sum())
    print("\n--- paired: causal EA minus no-recal (per subject) ---")
    for r, d in zip(results, diffs):
        print(f"  subject {r['subject']}: {d:+.3f}")
    print(f"mean diff: {diffs.mean():+.3f}  ({n_up}/{len(diffs)} subjects improve)")
    try:
        stat, p = wilcoxon(diffs)
        print(f"Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f} (two-sided)")
        wp = float(p); ww = float(stat)
    except ValueError as e:
        print(f"Wilcoxon not computable: {e}")
        wp = ww = None
    stats = {"mean_diff": float(diffs.mean()), "n_improve": n_up, "n": len(diffs),
             "wilcoxon_W": ww, "wilcoxon_p": wp, "per_subject_diff": diffs.tolist()}
    # Append stats as a trailing record so the JSON stays a list the plot reads.
    RESULTS.write_text(json.dumps(results + [{"paired_stats_ea_vs_norecal": stats}], indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
