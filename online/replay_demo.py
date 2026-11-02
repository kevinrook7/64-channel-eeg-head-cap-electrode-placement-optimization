"""Replay demo: stream a session as causal windows and show live left/right calls.

This is the honest "fake live loop". It does NOT replay whole cue-locked trials
(that would prove little). It walks each session-2 trial sample-block by
sample-block, decoding from a causal sliding window that only sees the past, and
prints / plots the running prediction as the imagery unfolds -- exactly what a
real device does with a live stream.

Data source: the lightweight .npz exports from export_for_colab.py (X, y,
ch_names, sfreq), so this runs with just numpy + the repo -- no MOABB / MNE load,
so it is safe on a low-memory host. The decoder (CSP+LR) is trained on session 1
(sub{N}_T.npz) and the stream is session 2 (sub{N}_E.npz).

Channels: selected on session 1 only (greedy on the full window), so no leakage.

Usage:
    python -m online.replay_demo --subject 1              # terminal readout
    python -m online.replay_demo --subject 1 --plot       # live matplotlib window
    python -m online.replay_demo --subject 1 --recal 20   # warm up on 20 session-2 trials first

Note the labels in these .npz files are 2 = left, 3 = right (export_for_colab.py
keeps the original BNCI codes; the separate Colab export remapped to 0/1).
"""

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "export_npz"
LABELS = {2: "LEFT ", 3: "RIGHT"}


def _covs(X):
    return np.einsum("nct,ndt->ncd", X, X) / X.shape[2]


def load_npz(data_dir, subject, session):
    d = np.load(Path(data_dir) / f"sub{subject}_{session}.npz", allow_pickle=True)
    return d["X"].astype(np.float64), d["y"].astype(np.int64), list(d["ch_names"]), float(d["sfreq"])


def select_channels(X, y, n_select, seed=0):
    """Greedy forward selection on this (session-1) data only."""
    from select_channels import greedy_forward
    selected, _ = greedy_forward(_covs(X), y, n_select, seed)
    return selected


def build_decoder(X, y, channels, seed=0):
    from fast_csp import make_cov_csp_decoder, subset_covariances
    covs = subset_covariances(_covs(X), channels)
    n = min(6, len(channels))
    dec = make_cov_csp_decoder(seed, n_components=n)
    dec.fit(covs, y)
    return dec


def causal_windows_for_trial(trial, sfreq, win_len, step, cue_offset):
    """Yield (latency_s, window_cov) as the trial 'streams' in, past-only windows.

    trial : (channels, times) for one epoch (starts cue_offset seconds before cue).
    """
    win = int(round(win_len * sfreq))
    hop = int(round(step * sfreq))
    cue = int(round(cue_offset * sfreq))
    n_times = trial.shape[1]
    stop = cue + win
    while stop <= n_times:
        start = stop - win
        w = trial[:, start:stop]
        cov = (w @ w.T) / w.shape[1]
        latency = stop / sfreq - cue_offset        # window end, s after cue
        yield latency, cov
        stop += hop


def run(subject, data_dir, n_select=8, win_len=2.0, step=0.25, recal=0, plot=False, seed=0):
    from fast_csp import subset_covariances

    Xtr, ytr, ch_names, sfreq = load_npz(data_dir, subject, "T")
    Xte, yte, _, _ = load_npz(data_dir, subject, "E")
    cue_offset = 0.5   # export window is -0.5..4.0 s around the cue

    selected = select_channels(Xtr, ytr, n_select, seed)
    print(f"subject {subject}: selected {n_select} channels on session 1:",
          ", ".join(ch_names[i] for i in selected))

    # Optional quick recalibration: add the first `recal` (balanced) session-2 trials.
    if recal > 0:
        per = recal // 2
        idx = []
        for c in (2, 3):
            idx.extend(np.where(yte == c)[0][:per].tolist())
        idx = np.array(sorted(idx))
        Xtr = np.concatenate([Xtr, Xte[idx]])
        ytr = np.concatenate([ytr, yte[idx]])
        mask = np.ones(len(yte), bool); mask[idx] = False
        Xte, yte = Xte[mask], yte[mask]
        print(f"recalibrated on {len(idx)} session-2 trials; {len(yte)} left to stream")

    dec = build_decoder(Xtr, ytr, selected, seed)

    if plot:
        _run_plot(dec, selected, Xte, yte, sfreq, win_len, step, cue_offset)
        return

    # Terminal readout: one trial at a time, prediction updating as it streams.
    correct = 0
    for t in range(len(Xte)):
        true = LABELS[yte[t]]
        last = None
        for latency, cov in causal_windows_for_trial(Xte[t], sfreq, win_len, step, cue_offset):
            sub = subset_covariances(cov[None], selected)
            pred = int(dec.predict(sub)[0])
            last = pred
            bar = "<<<<" if pred == 2 else ">>>>"
            print(f"\rtrial {t+1:3d}/{len(Xte)}  t+{latency:4.2f}s  "
                  f"pred {LABELS[pred]} {bar}   (true {true})", end="", flush=True)
            time.sleep(0.03)
        correct += (last == yte[t])
        print()
    print(f"\nfinal-window streaming accuracy: {correct/len(yte):.3f} on {len(yte)} trials")


def _run_plot(dec, selected, Xte, yte, sfreq, win_len, step, cue_offset):
    import matplotlib.pyplot as plt
    from fast_csp import subset_covariances

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-1, 1); ax.set_ylim(0, 1); ax.axis("off")
    arrow = ax.text(0, 0.6, "", ha="center", va="center", fontsize=40)
    label = ax.text(0, 0.25, "", ha="center", va="center", fontsize=14)
    correct = 0
    for t in range(len(Xte)):
        last = None
        for latency, cov in causal_windows_for_trial(Xte[t], sfreq, win_len, step, cue_offset):
            pred = int(dec.predict(subset_covariances(cov[None], selected))[0])
            last = pred
            arrow.set_text("<<" if pred == 2 else ">>")
            arrow.set_color("tab:blue" if pred == 2 else "tab:red")
            label.set_text(f"trial {t+1}/{len(Xte)}  t+{latency:.2f}s  "
                           f"pred {LABELS[pred].strip()}  (true {LABELS[yte[t]].strip()})")
            fig.canvas.draw(); fig.canvas.flush_events()
            plt.pause(0.03)
        correct += (last == yte[t])
    plt.ioff()
    print(f"final-window streaming accuracy: {correct/len(yte):.3f}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Causal replay demo for session-2 streaming.")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--channels", type=int, default=8, dest="n_select")
    p.add_argument("--win-len", type=float, default=2.0)
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--recal", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.subject, a.data_dir, a.n_select, a.win_len, a.step, a.recal, a.plot)
