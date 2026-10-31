"""Steps 3 + 4 rerun with Filter Bank CSP on BCI IV 2a (step 6).

Repeats the within-session (step 3, bnci_evaluate.py) and cross-session (step 4,
bnci_cross_session.py) channel-selection comparisons, but with FBCSP instead of
plain single-band CSP, to see whether greedy's edge over the motor strip / random
survives a stronger decoder.

Methods per size k (4/8/16), exactly as before:
  full    all 22 channels
  greedy  channels chosen by FBCSP greedy forward selection (training data only)
  motor   fixed motor-strip channels (no labels)
  random  random k-channel subsets, N_RANDOM draws (no labels)

Within-session: nested 5-fold outer CV per session, greedy chosen on the inner
training trials. Cross-session: channels chosen + FBCSP fit on day 1 (T), scored
once on all 144 day-2 (E) trials.

Leakage: band filters are fixed (not fit); FBCSP's CSP, MI feature selection and
LR are fit on training trials only; channel selection is inside the training
split. Same guarantees as the plain-CSP scripts.

Cost: FBCSP greedy is ~9x the single-band work. Subjects run in parallel but the
worker count is capped (N_JOBS) to bound peak memory on a tight host; band
covariances are built once per session and reused.
"""

import json
import time
from pathlib import Path

import mne
import numpy as np
from joblib import Parallel, delayed

from baseline import N_SPLITS, SEED, task_window
from bnci_evaluate import BNCI_SIZES, METHODS, SUBJECTS
from evaluate import N_RANDOM, mean_std, motor_subset
from fbcsp import FILTER_BANK, band_trial_covariances
from fbcsp_select import fb_greedy_forward, fb_test_accuracy, penalized_size
from load_bnci import prepare_bnci_sessions
from sklearn.model_selection import StratifiedKFold

N_JOBS = 3          # bounded: FBCSP holds 9 bands of covariances per worker
MAX_SIZE = max(BNCI_SIZES)


def _bands(epochs):
    return band_trial_covariances(task_window(epochs), epochs.info["sfreq"], FILTER_BANK)


def within_session_subject(subject, band_covs, y, names, seed=SEED):
    """FBCSP full / greedy / motor / random, within-session nested CV, for one session."""
    n_channels = band_covs[0].shape[1]
    all_ch = list(range(n_channels))
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    rng = np.random.default_rng([seed, subject, 6])
    idx = np.arange(len(y))

    res = {"full": [], "penalized": [], "pen_sizes": [],
           "greedy": {k: [] for k in BNCI_SIZES},
           "motor": {k: [] for k in BNCI_SIZES},
           "random": {k: [] for k in BNCI_SIZES}}

    for tr, te in outer.split(idx, y):
        tr_covs = [c[tr] for c in band_covs]
        te_covs = [c[te] for c in band_covs]
        y_tr, y_te = y[tr], y[te]

        selected, path = fb_greedy_forward(tr_covs, y_tr, MAX_SIZE, seed)
        k_pen = penalized_size(path)

        res["full"].append(fb_test_accuracy(tr_covs, y_tr, te_covs, y_te, all_ch, seed))
        res["penalized"].append(fb_test_accuracy(tr_covs, y_tr, te_covs, y_te, selected[:k_pen], seed))
        res["pen_sizes"].append(k_pen)
        for k in BNCI_SIZES:
            res["greedy"][k].append(fb_test_accuracy(tr_covs, y_tr, te_covs, y_te, selected[:k], seed))
            res["motor"][k].append(fb_test_accuracy(tr_covs, y_tr, te_covs, y_te, motor_subset(names, k), seed))
            draws = [sorted(rng.choice(n_channels, k, replace=False)) for _ in range(N_RANDOM)]
            res["random"][k].append(
                float(np.mean([fb_test_accuracy(tr_covs, y_tr, te_covs, y_te, d, seed) for d in draws]))
            )

    out = {"subject": subject, "n_channels": n_channels,
           "full": float(np.mean(res["full"])),
           "penalized": float(np.mean(res["penalized"])),
           "pen_sizes": res["pen_sizes"]}
    for m in METHODS:
        out[m] = {k: float(np.mean(res[m][k])) for k in BNCI_SIZES}
    return out


def cross_session_subject(subject, bc_t, y_t, bc_e, y_e, names, seed=SEED):
    """FBCSP cross-session: choose + fit on T, score once on all of E."""
    n_channels = bc_t[0].shape[1]
    picks_t, path = fb_greedy_forward(bc_t, y_t, MAX_SIZE, seed)
    k_pen = penalized_size(path)
    rng = np.random.default_rng([seed, subject, 7])

    def day2(channels):
        return fb_test_accuracy(bc_t, y_t, bc_e, y_e, list(channels), seed)

    out = {"subject": subject, "n_channels": n_channels,
           "full": day2(range(n_channels)),
           "penalized": day2(picks_t[:k_pen]), "penalized_size": k_pen,
           "greedy": {}, "motor": {}, "random": {}}
    for k in BNCI_SIZES:
        out["greedy"][k] = day2(picks_t[:k])
        out["motor"][k] = day2(motor_subset(names, k))
        draws = [sorted(rng.choice(n_channels, k, replace=False)) for _ in range(N_RANDOM)]
        out["random"][k] = float(np.mean([day2(d) for d in draws]))
    return out


def summarize(rows, title):
    sizes_txt = "/".join(str(k) for k in BNCI_SIZES)
    print(f"\n=== {title} ===")
    print("subj  full | " + " | ".join(f"{m} {sizes_txt}" for m in METHODS) + " | penalized")
    for r in rows:
        cells = " | ".join(" ".join(f"{r[m][k]:.2f}" for k in BNCI_SIZES) for m in METHODS)
        pen = f"{r['penalized']:.2f}"
        pen += f" ({r['penalized_size']}ch)" if "penalized_size" in r else f" ({r['pen_sizes']})"
        print(f"{r['subject']:>4}  {r['full']:.3f} | {cells} | {pen}")
    print(f"\nAcross {len(rows)} subjects (mean +/- std):")
    print(f"  all {rows[0]['n_channels']} channels: {mean_std([r['full'] for r in rows])}")
    for k in BNCI_SIZES:
        print(f"  {k:>2} ch  " + "   ".join(f"{m} {mean_std([r[m][k] for r in rows])}" for m in METHODS))
    full = np.array([r["full"] for r in rows])
    print("  paired (per subject, then mean +/- std):")
    for k in BNCI_SIZES:
        g, m, rnd = (np.array([r[x][k] for r in rows]) for x in METHODS)
        print(f"    {k:>2} ch  greedy - motor {mean_std(g - m)} (wins {int((g > m).sum())})"
              f"   greedy - random {mean_std(g - rnd)} (wins {int((g > rnd).sum())})")


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    start = time.time()

    print(f"FBCSP bands: {FILTER_BANK}")
    print("Loading sessions and building band covariances...")
    loaded = {}
    for subject in SUBJECTS:
        sessions = prepare_bnci_sessions(subject)
        loaded[subject] = {
            "T": (_bands(sessions["T"]), sessions["T"].events[:, 2]),
            "E": (_bands(sessions["E"]), sessions["E"].events[:, 2]),
            "names": sessions["T"].ch_names,
        }
        print(f"  subject {subject} ready ({time.time() - start:.0f}s)")

    saved = {}

    # Step 3: within-session, session T then E (each session scored on its own folds).
    for sess in ("T", "E"):
        jobs = [(s, *loaded[s][sess][:2], loaded[s]["names"]) for s in SUBJECTS]
        rows = Parallel(n_jobs=N_JOBS)(delayed(within_session_subject)(*j) for j in jobs)
        summarize(rows, f"FBCSP within-session, session {sess}")
        saved[f"within_{sess}"] = rows

    # Step 4: cross-session T -> E.
    jobs = [(s, *loaded[s]["T"], *loaded[s]["E"], loaded[s]["names"]) for s in SUBJECTS]
    rows = Parallel(n_jobs=N_JOBS)(delayed(cross_session_subject)(*j) for j in jobs)
    summarize(rows, "FBCSP cross-session, train T -> test E")
    saved["cross_TE"] = rows

    # Persist per-subject rows so figures can be drawn later without re-running
    # this ~67 min job. Numbers only (no raw/derived EEG), so safe to keep.
    RESULTS_DIR = Path(__file__).resolve().parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "bnci_fbcsp_results.json"
    with open(out_path, "w") as f:
        json.dump({"sizes": BNCI_SIZES, "results": saved}, f, indent=2)
    print(f"\nSaved per-subject results to {out_path}")

    print(f"\nTotal time: {time.time() - start:.0f}s")
