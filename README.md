# MyEEG

**Can a few well-chosen EEG electrodes decode motor imagery as well as a full cap — and keep working day to day?**

This project studies personalized, task-aware EEG channel selection for brain-computer interfaces, on left- vs right-hand motor imagery. It is software only: it works from public, already-recorded EEG and asks two questions in order.

- **Part 1 — channel selection.** Can a small, personalized channel subset match a full-cap decoder? Answer: **yes for accuracy, no for stability.** Eight selected channels match the 22-channel accuracy, but the *best* channels drift from one session (day) to the next.
- **Part 2 — online decoding and quick recalibration.** Since the best channels drift, can a small channel set fixed on day 1 be *recalibrated quickly* on day 2 and still decode **causally, online**? This part adds a sliding-window causal decoder, a recalibration-vs-trials sweep, and a Euclidean-alignment transfer baseline.

Two classes only. BCI IV 2a has four (left hand, right hand, feet, tongue); everything here uses the **left/right two-class subset**, and that is stated wherever a number is reported.

## Headline result

**TL;DR (n=9, causal, fixed held-out session-2 trials): free unsupervised Euclidean alignment reaches 0.757 vs 0.737 with no recalibration — above a 40-trial labeled warm-up (0.748), for zero labels. A small mean gain with mixed per-subject direction (p = 0.148): suggestive, not established.**

![Accuracy vs recalibration trials and vs decision latency](results/online_recalibration.png)

*(Part 2 figure — produced by `online/run_online.py` then `online/plot_online.py`. See "Reproduce Part 2". Panel 1: all conditions on one fixed held-out set. Panel 3: the rolling EA−no-recal difference stays flat near zero, so EA's edge is constant, not a reference that stabilizes. Panel 6: the pre-cue control decodes pure rest at chance, so nothing leaks backward. Shaded bands in panels 2–4 are ±1 SD across the 9 heterogeneous subjects, so they are wide; panel 5 is the cleaner per-subject view of the EA effect.)*

The settled Part 1 figure (accuracy vs channel count, cross-session) is `results/bnci_fbcsp.png`.

---

## Part 1 — channel selection (frozen as tag `v1-channel-selection`)

The exact v1 numbers live in [`docs/results-summary.md`](docs/results-summary.md) and `results/`. Part 2 did add a `causal=` argument to `preprocess.filter_and_flag` / `load_bnci.prepare_bnci_sessions`, but it **defaults to `False` (the original zero-phase filter)** and every Part 1 script calls these with no `causal` argument, so Part 1 behavior is bit-identical to v1 — a test (`tests/test_online.py::test_filter_defaults_to_noncausal_zero_phase`) asserts the default path is unchanged and differs from the causal one. These numbers do not silently change; `git checkout v1-channel-selection` reproduces the original state.

Key findings on BCI IV 2a (9 subjects, 2 sessions on different days, 144 left/right trials/session, 22 channels), FBCSP + CSP decoder, no leakage (channels and models fit on training trials only):

- **8 channels match 22.** Cross-session (train day 1, test day 2): 8ch 0.777 vs all-22 0.772; 16ch 0.796; 4ch 0.761. Fewer channels barely cost accuracy.
- **Best channels drift.** Greedy picks on day 1 and day 2 overlap no better than chance, and the region test shows the drift is geometric, not just neighbour-swapping. So a fixed-forever placement degrades — the design answer is fixed electrodes over the motor region with redundancy, plus a decoder re-trained each session.
- Deep models (EEGNet, DANN, pooled) were tried on Colab and **all lost to CSP** (0.721 / 0.718 / 0.728 vs 0.772); see `docs/results-summary.md`.

Scripts: `baseline.py`, `select_channels.py`, `fast_csp.py`, `fbcsp.py`, `fbcsp_select.py`, `bnci_*.py`. Figures: `results/bnci_fbcsp.png`, `results/bnci_cross_session.png`, `results/bnci_region_test.png`.

### Reproduce Part 1

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# BCI IV 2a downloads automatically via MOABB into data/ on first run (see "Data").
python bnci_fbcsp.py          # ~67 min; writes results/bnci_fbcsp_results.json
python bnci_fbcsp_plot.py     # draws results/bnci_fbcsp.png from that JSON
python -m pytest tests/ -v    # no-leakage + CovCSP-matches-MNE checks
```

---

## Part 2 — online decoding and quick recalibration (`online/`)

New work, kept in its own package so the Part 1 numbers above never move. It answers: *given that the best channels drift, can a day-1 channel set recover on day 2 with a short, labeled warm-up, decoding causally and online?*

**Honest-evaluation contract** (enforced by `tests/test_online.py`):

- **Causal.** A decision at time *t* uses only samples at or before *t*. A sliding window looks backward only; future samples are never read. Accuracy is reported as a function of **decision latency** (time since cue).
- **No test-session leakage.** Channels are selected on **session 1 only**; session 2 is the stream.
- **Recalibration is held out.** The *k* labeled recalibration trials are removed from the test set, so no trial is both trained on and scored.
- **Transfer baseline uses no labels and is causal.** Euclidean alignment (He & Wu, 2020) aligns sessions using only the **unlabeled** incoming covariances, with the test reference a **running mean over trials seen so far** (never future trials) — online, not transductive.

Modules:

- `online/causal_decoder.py` — causal sliding-window CSP+LR; accuracy vs decision latency.
- `online/euclidean_alignment.py` — Euclidean-alignment transfer baseline (zero labels).
- `online/recalibration.py` — accuracy vs number of recalibration trials (0/5/10/20/40), three conditions: no-recal, EA, quick recalibration.
- `online/run_online.py` — runs both sweeps over 9 subjects, measures per-decision compute latency, writes `results/online_results.json`.
- `online/plot_online.py` — draws `results/online_recalibration.png` from that JSON (recomputes nothing).
- `online/replay_demo.py` — causal replay demo: streams session 2 window by window and shows live left/right predictions.

### Results

Cross-subject means (9 subjects, 8 channels selected on session 1, left/right two-class, held-out session 2). The band-pass filter is **causal** (minimum-phase FIR, `phase='minimum'`), so every windowed decision uses only samples at or before the decision time — see "Causal filter" below. The headline decision uses a **causal 2.0 s window ending 2.0 s after the cue**. Numbers are generated straight from `results/online_results.json` by `python -m online.print_readme_numbers` (so this table is never transcribed by hand); the figure is `results/online_recalibration.png`.

**Everything is scored on one fixed held-out set** — session-2 trials 41–144 (104 trials), the trials after the largest warm-up — so no-recalibration, causal EA, quick recalibration and the combo all see the *identical* test trials at every budget. The *k* recalibration trials are the **first k of session 2 in stream order** (a warm-up at the start of the day, never a trial from later than the ones it is scored against). Causal EA's running reference also starts at trial 0, so by the time scoring begins at trial 41 it has seen 40 unlabeled trials — fair, and the same amount of stream the k=40 warm-up used.

| condition (fixed 104-trial held-out set) | mean accuracy (n=9) |
|---|---|
| no recalibration (session-1 decoder as-is) | 0.737 |
| **Euclidean alignment — causal** (running reference, **0 labels**) | **0.757** |
| quick recalibration, 5 labeled trials | 0.733 |
| quick recalibration, 10 labeled trials | 0.746 |
| quick recalibration, 20 labeled trials | 0.740 |
| quick recalibration, 40 labeled trials | 0.748 |
| causal EA + recalibration, 40 labeled trials | 0.753 |

(The full-stream causal-EA number, scored on all 144 trials for the latency/paired analyses, is 0.760; on the fixed 104-trial set it is 0.757. The two differ only by which trials are scored.)

**The fair comparison is cleaner than the quick-look suggested:** on identical trials, label-free causal EA (0.757) beats quick recalibration at *every* budget tested, including 40 labeled trials (0.748). So "alignment can substitute for calibration trials" holds up when scored apples-to-apples — the labeled warm-up does not catch EA even at k=40.

Decode latency: **0.6 ms per window, end to end** — covariance from the raw window samples, the channel-subset slice, CSP log-power, and the logistic-regression call. It excludes the one-off per-session band-pass filter (done once up front, not per window). Either way the decoder is far faster than real time, so latency is set by how much post-cue EEG you wait for, not by compute.

**Pre-cue control (no backward leakage).** A causal filter still has to be shown not to smear imagery back into the pre-cue window. So the sweep decodes a 0.5 s window that ends *at* the cue — pure resting EEG, no imagery yet. Across 9 subjects it comes out at **mean 0.488, every subject near chance** (figure panel 6). Nothing leaks backward: an above-chance "pre-cue" decode would have meant contamination, and there is none.

**Euclidean alignment is causal here.** The headline EA number whitens each test trial by the running mean of the test covariances *seen so far* (trials 0..t, `online.euclidean_alignment.streaming_reference`), never future trials — so it is online, not transductive. The transductive variant (one reference from *all* test trials) is 0.755 — essentially the same as the causal 0.760 — so the finding does not rest on the transductive shortcut.

### Causal filter

The 8–30 Hz band-pass is applied once per session with a **minimum-phase (causal) FIR** (`preprocess.filter_and_flag(causal=True)`, MNE `phase='minimum'`), so filtered sample *t* depends only on samples at or before *t*. This matters because the online claim is "a decision at time *t* uses only data up to *t*": a zero-phase (forward–backward) filter — MNE's default, and what Part 1's offline cue-locked decode uses — would violate that by letting later samples shape earlier ones. The Part 1 offline scripts keep the default zero-phase filter (the whole window is seen at once, so it is not a causality claim); only the online sweeps switch to the causal filter. Switching from zero-phase to minimum-phase moved the numbers only slightly (see the comparison table below), did **not** inflate the early-latency result, and the pre-cue control confirms no backward leakage remains.

**Findings:**

- **Free, online, unsupervised transfer beats a labeled warm-up, by a small margin.** On the identical held-out set, causal Euclidean alignment reaches 0.757 with **zero labels**, above quick recalibration at every budget including 40 labeled trials (0.748) and above no-recalibration (0.737). But the per-subject effect is mixed: causal EA minus no-recal is **5 up, 1 tie, 3 down; Wilcoxon on the 8 non-tied subjects, mean +0.017, p = 0.148** — so the honest reading is "causal EA gives a small mean gain (0.737 → 0.757) with mixed per-subject direction, suggestive but not established at n = 9." It is not a proven win; it is evidence that alignment can stand in for a short labeled warm-up. Combining EA with a warm-up (causal EA + 40 trials, 0.753) adds nothing over EA alone.
- **Longer windows decode better; earlier decisions are still better — and this survives the causal filter.** The window-length sweep (0.5 / 1.0 / 1.5 / 2.0 s) shows a longer window beats a shorter one at every latency, and within any fixed window length accuracy falls as the window ends later. The cross-subject latency curve peaks at the earliest clean window (1.5 s, 0.743) and declines monotonically to 0.621 at 3.5 s. This was the reviewer's main worry: that the non-causal filter was faking the early advantage by smearing later imagery back into early windows. **It is not the filter** — the effect is the same under the causal filter — and it is **not leakage** — the pre-cue control is at chance. So the drop with latency is a real property of these trials (the discriminative imagery is early), with cue-locked visual/eye activity the one explanation the data here cannot separate from genuine early imagery. The 8 selected channels track all 22 closely across the latency curve.
- **A short window at short latency can dip into pre-cue baseline.** With a 2.0 s window, a decision at 1.5 s latency would start 0.5 s *before* the cue and mix in resting EEG; the sweep drops such windows, so every plotted point is a clean post-cue decision.
- **Online decoding is real-time.** 0.48 ms end-to-end per window.
- **Same hard-subject split as Part 1.** Subjects 3/8/9 decode ~0.89–0.95; subjects 2/5/6 sit near chance (~0.51–0.59), and none of the methods tried rescues them — consistent with the deep-model result that the limit is signal, not method.

Honest limits: one dataset (BCI IV 2a), 9 subjects, two classes. The EA gain over no-recalibration is consistent in direction but not significant at this sample size, and weaker under the causal filter than under zero-phase. **Causal EA's small edge does not come from the reference "locking in" over the stream.** Figure panel 3 plots the **rolling 20-trial difference** between causal EA and no-recalibration on the same trials: if the reference were stabilizing, that difference would rise as more unlabeled trials arrive. Instead it hovers just above zero (≈ +0.02) flat across the whole session — so EA runs a roughly constant small step ahead, and the earlier cumulative-accuracy "peak near trial 18" was trial difficulty, not the reference stabilizing. A 2.0 s window also cannot report latencies whose window would start before the cue (dropped in the sweep).

### Non-causal vs causal (what the filter change cost)

The online sweeps originally used MNE's default zero-phase (non-causal) band-pass, which quietly violated the causality claim. Switching to the causal `phase='minimum'` filter moved the headline numbers only slightly, and — importantly — did **not** remove the early-latency advantage, confirming that result was not a filter artifact:

| metric | non-causal (zero-phase) | causal (minimum-phase) |
|---|---|---|
| no recalibration | 0.728 | 0.743 |
| causal EA (headline, full stream) | 0.754 | 0.760 |
| accuracy at 1.5 s latency (earliest) | 0.744 | 0.743 |
| accuracy at 3.5 s latency (latest) | 0.633 | 0.621 |
| EA − no-recal, paired | 7 up / 1 tie / 1 down, +0.026, p = 0.078 | 5 up / 1 tie / 3 down, +0.017, p = 0.148 |
| pre-cue control | not run | 0.488 (chance, no leak) |

The earliest-latency advantage (1.5 s > 3.5 s) is present under both filters, so it is a property of the data, not the filter. The full non-causal result set is archived at [`docs/online_results_noncausal.json`](docs/online_results_noncausal.json); the live `results/online_results.json` is the causal one.

### Reproduce Part 2

```powershell
.\.venv\Scripts\Activate.ps1
python -m online.run_online        # writes results/online_results.json (all 9 subjects)
python -m online.plot_online       # draws results/online_recalibration.png from the JSON
python -m online.print_readme_numbers   # prints the exact table above from the JSON
```

Replay demo (runs from the lightweight `.npz` exports — no MOABB load needed):

```powershell
python export_for_colab.py                 # one-time: writes export_npz/sub*_{T,E}.npz
python -m online.replay_demo --subject 1          # terminal: streaming left/right calls
python -m online.replay_demo --subject 1 --plot   # live arrow window
python -m online.replay_demo --subject 1 --recal 20   # warm up on 20 session-2 trials first
```

---

## Data

BCI Competition IV dataset 2a (BNCI2014_001). It is **not redistributed here** — it downloads automatically on first run via MOABB into the gitignored `data/` folder (the loader sets `MNE_DATASETS_BNCI_PATH` to `data/` for this process only). Dataset home: <https://www.bbci.de/competition/iv/#dataset2a>, served through MOABB (<https://neurotechx.github.io/moabb/>). The 9 subjects × 2 sessions are ~1.6 GB on first download.

The older PhysioNet pilot (`load_data.py`, EEG Motor Movement/Imagery, <https://physionet.org/content/eegmmidb/1.0.0/>) decoded clearly on only 3 of 10 subjects and is kept for history; BCI IV 2a is the dataset the results rest on.

## Environment

Pinned in [`requirements.txt`](requirements.txt): Python 3.12, `mne`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `moabb`, `pytest`. `pylsl` is only for the optional real-time LSL stub (`streamer.py`). No GPU is needed for any of Part 1 or Part 2; the Colab deep-model work (`colab_deep_model.py`) is the only GPU piece and is separate.

## Docs

- [`docs/results-summary.md`](docs/results-summary.md): full Part 1 + deep-model results with every number.
- [`docs/deep-model-plan.md`](docs/deep-model-plan.md): the parked deep-model (drift-tolerance) plan.
- [`AGENTS.md`](AGENTS.md): project vision, research question, pipeline, evaluation rules.
