"""Region test: do day-1 and day-2 greedy picks cover the same scalp area? (step 5)

Background (from step 4, bnci_cross_session.py): greedy run on session T and on
session E picks *different electrodes* -- set overlap is at chance level. But
neighbouring electrodes are highly correlated (e.g. C5 ~ C3 at 0.92), so greedy
choosing one neighbour over another may be noise while the *region* is stable.

This test measures that directly, using scalp geometry, not accuracy:

  - For each subject, get the greedy pick order on T and on E (same procedure as
    bnci_cross_session.py: greedy_forward on each session's own covariances).
  - Get the 22 channels' 3D positions from the colin27_1005 montage (metres).
  - For k = 4, 8, 16: for each of E's k picks, distance to its nearest T pick;
    average over the k picks. Small = E's picks sit near T's picks on the scalp.
  - Null: replace E's k picks with 1000 random k-channel draws and recompute the
    same nearest-T-pick distance. p = fraction of draws whose mean distance is
    <= the real one. A small p means E lands closer to T than random channels do.

NO LEAKAGE: this test scores zero accuracy and fits no model. It only compares
*where* picks land on the scalp. The picks come from greedy_forward, which already
uses each session's own trials; comparing their 3D positions touches no held-out
labels and trains nothing. So there is no train/test split to protect here.

Distances are reported in centimetres. ~1 min on 12 cores (greedy on both
sessions, no scoring).
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window
import matplotlib.pyplot as plt
import mne
import numpy as np
from joblib import Parallel, delayed  # installed with scikit-learn

from baseline import SEED, task_window
from bnci_evaluate import BNCI_SIZES, SUBJECTS
from evaluate import RESULTS_DIR, mean_std
from fast_csp import trial_covariances
from load_bnci import prepare_bnci_sessions
from select_channels import greedy_forward

N_NULL = 1000  # random k-channel draws for the null distribution


def channel_positions_cm(epochs):
    """(n_channels, 3) array of each channel's 3D position in cm, montage order.

    Positions come from the colin27_1005 montage that preprocess.filter_and_flag
    already attached. MNE stores them in metres; we return centimetres so the
    distances read in familiar scalp units.
    """
    ch_pos = epochs.get_montage().get_positions()["ch_pos"]
    return np.array([ch_pos[name] for name in epochs.ch_names]) * 100.0


def nearest_pick_distance(pos, e_picks, t_picks):
    """Mean distance (cm) from each E pick to its nearest T pick.

    pos: (n_channels, 3) positions. e_picks / t_picks: channel index lists.
    For every E pick, take the smallest Euclidean distance to any T pick, then
    average over the E picks. Lower = E picks sit closer to the T picks.
    """
    e_xyz = pos[np.asarray(e_picks)]            # (k, 3)
    t_xyz = pos[np.asarray(t_picks)]            # (k, 3)
    d = np.linalg.norm(e_xyz[:, None, :] - t_xyz[None, :, :], axis=2)  # (k, k)
    return float(d.min(axis=1).mean())


def region_test_subject(subject, c_t, y_t, c_e, y_e, pos, names, seed=SEED):
    """Greedy picks on T and E, then the nearest-pick distance and its null per k."""
    start = time.time()
    n_channels = c_t.shape[1]
    max_size = max(BNCI_SIZES)

    picks_t, _ = greedy_forward(c_t, y_t, max_size, seed)   # day 1
    picks_e, _ = greedy_forward(c_e, y_e, max_size, seed)   # day 2
    rng = np.random.default_rng([seed, subject, 5])

    out = {
        "subject": subject,
        "n_channels": n_channels,
        "real": {}, "null_mean": {}, "p": {},
        "picks_t": {}, "picks_e": {},
    }
    for k in BNCI_SIZES:
        t_k, e_k = picks_t[:k], picks_e[:k]
        real = nearest_pick_distance(pos, e_k, t_k)
        draws = [rng.choice(n_channels, k, replace=False) for _ in range(N_NULL)]
        null = np.array([nearest_pick_distance(pos, d, t_k) for d in draws])
        out["real"][k] = real
        out["null_mean"][k] = float(null.mean())
        # One-sided: is the real distance smaller than random picks' distance?
        out["p"][k] = float((null <= real).mean())
        out["picks_t"][k] = [names[i] for i in t_k]
        out["picks_e"][k] = [names[i] for i in e_k]
    out["seconds"] = time.time() - start
    return out


def plot_picks_on_scalp(rows, epochs, path, k=8):
    """Scalp map per subject: T picks (circles) vs E picks (crosses), 2D topo layout."""
    layout = epochs.get_montage().get_positions()["ch_pos"]
    xy = {name: layout[name][:2] for name in epochs.ch_names}  # top-down x, y
    name_to_idx = {name: i for i, name in enumerate(epochs.ch_names)}

    n = len(rows)
    cols = 3
    panels = (n + cols - 1) // cols
    fig, axes = plt.subplots(panels, cols, figsize=(3.2 * cols, 3.2 * panels))
    axes = np.atleast_1d(axes).ravel()

    all_x = np.array([p[0] for p in xy.values()])
    all_y = np.array([p[1] for p in xy.values()])
    for ax, r in zip(axes, rows):
        ax.scatter(all_x, all_y, s=8, color="lightgray")  # all 22 channels
        t_names = set(r["picks_t"][k])
        e_names = set(r["picks_e"][k])
        for name in t_names:
            x, y = xy[name]
            ax.scatter(x, y, s=140, facecolors="none", edgecolors="tab:blue", linewidths=1.8)
        for name in e_names:
            x, y = xy[name]
            ax.scatter(x, y, marker="x", s=70, color="tab:red", linewidths=1.8)
        ax.set_title(f"S{r['subject']}  d={r['real'][k]:.1f}cm p={r['p'][k]:.2f}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.suptitle(f"BCI IV 2a greedy picks, k={k}: day 1 (blue o) vs day 2 (red x)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    start = time.time()

    jobs, montage_epochs = [], None
    for subject in SUBJECTS:
        sessions = prepare_bnci_sessions(subject)   # same channel set in T and E
        t, e = sessions["T"], sessions["E"]
        if montage_epochs is None:
            montage_epochs = t  # positions identical across subjects (template montage)
        pos = channel_positions_cm(t)
        jobs.append((subject, trial_covariances(task_window(t)), t.events[:, 2],
                     trial_covariances(task_window(e)), e.events[:, 2], pos, t.ch_names))

    rows = Parallel(n_jobs=-1)(delayed(region_test_subject)(*job) for job in jobs)

    print("\nRegion test: distance from each day-2 greedy pick to its nearest day-1 pick")
    print("(centimetres; lower = day-2 picks sit near day-1 picks. NO accuracy scoring, no leakage.)")
    print(f"null = {N_NULL} random k-channel draws vs the same day-1 picks; "
          "p = fraction of draws at least as close as the real day-2 picks")
    sizes_txt = BNCI_SIZES
    print("\nsubj | " + " | ".join(f"k={k}: real null p" for k in sizes_txt))
    for r in rows:
        cells = " | ".join(f"{r['real'][k]:4.1f} {r['null_mean'][k]:4.1f} {r['p'][k]:.3f}"
                           for k in sizes_txt)
        print(f"{r['subject']:>4} | {cells}")

    print(f"\nAcross {len(rows)} subjects (mean +/- std):")
    for k in sizes_txt:
        real = [r["real"][k] for r in rows]
        null = [r["null_mean"][k] for r in rows]
        beats = sum(r["p"][k] < 0.05 for r in rows)
        closer = sum(r["real"][k] < r["null_mean"][k] for r in rows)
        print(f"  k={k:>2}: real {mean_std(real)} cm   null {mean_std(null)} cm   "
              f"closer than null in {closer}/{len(rows)}   p<0.05 in {beats}/{len(rows)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    figure = RESULTS_DIR / "bnci_region_test.png"
    plot_picks_on_scalp(rows, montage_epochs, figure, k=8)
    print(f"\nSaved {figure}")
    print(f"Total time: {time.time() - start:.0f}s")
