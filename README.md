# MyEEG

**Personalized EEG electrode selection for brain-computer interfaces.**

MyEEG asks a simple research question: *can a personalized, task-aware channel-selection optimizer decode intended movements (starting with left- vs right-hand motor imagery) as accurately as a full-channel baseline while using fewer EEG electrodes?*

The first phase is software only: pick channel subsets from public, already-recorded EEG (the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)) and compare them against an all-channel baseline using leakage-safe, subject-specific evaluation. Physical placement recommendations and reconfigurable hardware come later, and only if the software results justify them.

## Status

Pipeline so far (Phases 1–4 on subject 1, Phase 5 on subjects 1–10):

- `load_data.py`: `load_subject()` downloads runs 4, 8, and 12, renames T0/T1/T2 per run before concatenating, and returns `raw, events, event_id`.
- `preprocess.py`: attaches a montage, band-pass filters 8–30 Hz, flags and removes bad channels, and cuts left/right epochs. `prepare_epochs()` returns those epochs.
- `baseline.py`: log band power (0.5–4.0 s) and L2 logistic regression, scored with stratified 5-fold cross-validation (seed 0). Subject 1, within-session: accuracy 0.556 ± 0.079 vs 0.511 for always guessing the larger class, so no evidence it beats chance yet. Common spatial patterns (CSP, 6 filters fit inside each training fold) + logistic regression on the same folds: 0.800 ± 0.183.

- `select_channels.py`: greedy forward channel selection inside nested cross-validation (channels chosen on training trials only). Subject 1, within-session: 8 channels 0.644 ± 0.183, 16 channels 0.733 ± 0.099, 32 channels 0.644 ± 0.183, vs 0.800 ± 0.183 for all 64. Fewer channels lost accuracy here, and the chosen channels changed from fold to fold.

- `fast_csp.py`: CSP computed from per-trial covariance matrices. Same fold scores as MNE's CSP, and makes greedy selection several times faster.
- `evaluate.py`: compares all channels, greedy subsets, a fixed motor-strip set (C3/C4/Cz outward), and random subsets on subjects 1–10, within-session. All channels 0.589 ± 0.235 across subjects; at 8 channels greedy 0.564, motor strip 0.587, random 0.541. Greedy did not beat the fixed motor strip, and only 3 of 10 subjects decode clearly above chance. Plot: `results/accuracy_vs_channels.png`.

- `load_bnci.py`: loads BCI Competition IV 2a (9 subjects, 2 sessions on different days, 144 left/right trials per session, 22 channels) through MOABB. `prepare_bnci_sessions()` gives both sessions the same channel set for cross-session tests.
- The bad-channel check only drops flat or unusually noisy channels; quiet channels are kept, since they can still carry real signal.

Learning-blueprint Phases 1–5 are now implemented for left vs right imagery. - `bnci_baseline.py`: all-channel baseline on BCI IV 2a, 9 subjects, within-session. CSP 0.773 ± 0.139 on session T and 0.806 ± 0.163 on session E (band power 0.692 / 0.707). 7 of 9 subjects decode well, unlike PhysioNet where only 3 of 10 did.

- `bnci_evaluate.py`: channel selection on BCI IV 2a, within-session, 4/8/16 channels. 16 channels lose only 1–2 points vs all 22. At 4 channels greedy beats the fixed motor strip by 2–4 points on average (clearly only on session E, 8 of 9 subjects), and random subsets by 5–7. Plots: `results/bnci_accuracy_vs_channels_T.png`, `..._E.png`.

- `bnci_cross_session.py`: channels chosen and decoders trained on day 1, tested on day 2. All 22 channels drop from 0.806 (within-session) to 0.741. At 4 channels, day-1 greedy picks score 0.725 vs 0.698 for the motor strip (higher in 7 of 9 subjects) and 0.663 for random (8 of 9). But greedy run on each day separately picks different channels, with overlap no better than chance, so the "best" channels for a person aren't stable from day to day here. Plot: `results/bnci_cross_session.png`.

- `bnci_region_test.py`: tests whether the day-1 and day-2 greedy picks at least cover the same scalp *region*, since neighbouring electrodes are highly correlated. For each subject it measures the mean distance from each day-2 pick to its nearest day-1 pick (3D montage positions, cm) against 1000 random channel sets. The picks land only slightly closer than random (k=4: 3.6 vs 4.5 cm, closer in 7 of 9 subjects but p<0.05 in only 2; k=8: 2.6 vs 2.7 cm, 1 of 9; k=16: 0.9 vs 1.0 cm, 0 of 9), so the day-to-day instability is geometric, not just a choice between neighbours. No accuracy is scored, so there is no leakage. Plot: `results/bnci_region_test.png`.

- `bnci_reverse_session.py`: reruns the cross-session test the other way (channels chosen and decoders trained on day 2, tested once on all 144 day-1 trials), to double the cross-session evidence. Greedy beats random and the motor strip in this direction too: at 4 channels greedy 0.705 vs motor 0.683 vs random 0.647 (greedy above random in 6 of 9 subjects), matching day 1 -> day 2 (0.725 / 0.698 / 0.663). So the accuracy claim holds in both directions, while the stability claim does not (region test). Reuses step 4's `cross_session_subject` with the sessions swapped; no leakage.

- `fbcsp.py`: Filter Bank CSP (Ang et al., 2008) built on the existing `CovCSP`. Nine 4 Hz bands from 4-40 Hz, CSP in each band, mutual-information feature selection (training only), then logistic regression. Within-session on subject 1 it beats plain CSP (session T 0.888 vs 0.827, session E 0.958 vs 0.916). `fbcsp_select.py`: FBCSP twins of the greedy selection functions (same algorithm, list-of-bands input), kept separate so the plain-CSP path is untouched.
- `bnci_fbcsp.py`: reruns within-session and cross-session channel selection with FBCSP. Accuracy rises a few points (cross-session all 22ch 0.772 vs 0.741 with plain CSP). Greedy's edge survives the stronger decoder and, cross-session, grows: at 4 channels greedy 0.761 vs motor 0.692 vs random 0.676 (greedy − motor +0.069, higher in 8 of 9 subjects, vs +0.027 with plain CSP). Within-session the gap still closes by 8-16 channels. FBCSP does not fix day-to-day stability (that is the region test). No leakage: band filters fixed; CSP, feature selection and LR fit on training trials only. ~67 min for the full rerun; per-subject results are saved to `results/bnci_fbcsp_results.json`. Figure: `results/bnci_fbcsp.png` — three panels (within day 1, within day 2, cross-session) of greedy/motor/random vs channel count with the all-22 point, drawn from that JSON by `bnci_fbcsp_plot.py` (recomputes nothing). Within-session the curves confirm greedy leads only at 4 channels and the methods converge by 16; cross-session greedy stays on top at every count.

Existing scripts from the original blueprint:

- `streamer.py`: a working simulated 4-channel EEG stream over LSL (untested). It's an optional extension for testing real-time infrastructure; simulated data can't show real BCI performance.
- `detector.py`: a stub (TODOs only) for LSL band power and a focus score.
- `agent.py`: a stub (TODOs only) for a contextual-bandit recommender. It's an optional extension.

See the status table in [`AGENTS.md`](AGENTS.md#11-status).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python streamer.py   # broadcasts the "fakeEEG" LSL stream until Ctrl+C
```

## Docs

- [`AGENTS.md`](AGENTS.md): the project vision, research question, pipeline, evaluation rules, and guidance for AI coding agents.
- [`docs/learning-blueprint.md`](docs/learning-blueprint.md): the Build-and-Deconstruct learning approach, reframed for this direction.

## Planned stack

Python, MNE-Python, NumPy/SciPy, scikit-learn. LSL (`pylsl`) only for optional real-time work.


Notes:

Look into usage of PyTorch
Look into Dr. Kai Keng Ang Filter Bank Common Spatial pattern Algorithm, FBCNet and GANs