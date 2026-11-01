# Deep-model plan (parked): fighting day-to-day drift on BCI IV 2a

Run this on **Google Colab** (free NVIDIA GPU). This laptop has only Intel Iris
Xe integrated graphics, which PyTorch cannot train on at usable speed. Nothing
here runs on this machine; it is a plan to pick up when a GPU is available.

## The question this step answers

Level 1 (done, see `results-summary.md`) found: a few well-chosen channels
decode as well as all 22, but **the best channels drift day to day** — there is
no stable personal montage. A fixed few-channel headset therefore needs
recalibration every session.

**This step's question:** can a deep model *tolerate* that drift — keep its
day-1 accuracy on day 2 better than CSP does — so the recalibration shrinks?

It is NOT trying to find a stable fixed montage (Level 1 ruled that out). It is
trying to find a *decoder* that does not care that the best spot moved.

## The number to beat

The honest baseline is the committed cross-session CSP/FBCSP result, train day 1
-> test day 2, mean over 9 subjects (full 22 channels):

| decoder | cross-session accuracy (all 22ch) |
| --- | --- |
| plain CSP | 0.741 |
| FBCSP | 0.772 |

A deep model is only interesting here if it **beats ~0.772 cross-session**, or
matches it with **far less per-session calibration data**. Within-session
accuracy is not the point — the drift gap is.

## Data

Same dataset, same preprocessing as this repo, so results are comparable:

- **BCI IV 2a**, 9 subjects, 2 sessions (T = day 1, E = day 2), 22 EEG channels,
  left vs right hand, 144 trials/session.
- Loader: `load_bnci.py`. Preprocessing: `preprocess.py` — band-pass **8–30 Hz**,
  epochs **−0.5 to 4.0 s** around the cue (codes 2 = left, 3 = right), bad-channel
  flagging. Keep this identical so the deep model and CSP see the same signal.
- Deep nets usually want the **raw filtered time series** (trials × channels ×
  times), not covariances. `preprocess.task_window()` already returns that shape.

**Getting data into Colab:** export the preprocessed arrays once on this machine
(`np.savez` of X/y per subject/session), upload the `.npz` to Google Drive, and
`drive.mount` in Colab. This avoids installing MNE + downloading the dataset on
Colab every session.

## Models, simplest first

### 1. FBCNet (baseline deep model)
A published motor-imagery CNN (filter-bank -> spatial conv -> temporal variance
-> classifier). Train on day 1, test on day 2, per subject. First milestone:
does it even match FBCSP cross-session? If not, stop — the data is too small.

### 2. Domain adaptation (the actual drift fix)
Add an adversarial or MMD term so day-1 and day-2 feature distributions are
forced to look alike, using day-2 trials **without labels** during training.
This is the part aimed squarely at drift. Expected to help most.

### 3. GAN-augmented training (most fragile)
Train a GAN to synthesize extra "sessions" with small shifts, train the decoder
on real + synthetic. Try only if 1–2 show promise; with 9 subjects a GAN can
hallucinate and hurt.

## Protocol (keep it honest)

- **Leakage:** test-session labels never seen in training; domain adaptation may
  use test-session *signals* (unlabeled) only, which is the standard
  unsupervised-DA setting — state this explicitly, it is the one subtle point.
- **Per subject, both directions** (T->E and E->T), mean ± std over 9 subjects,
  exactly as the CSP work reports, so the comparison is apples-to-apples.
- **Seed everything**; report variance across seeds — small data means noisy runs.
- **Checkpoint to Drive** every few epochs: Colab free sessions disconnect after
  a few hours, so a long run must survive a drop.

## Honest expectations

- With **9 subjects**, deep models may **not** beat CSP. That negative result is
  itself worth reporting ("even a strong model can't fix drift with this little
  data").
- The realistic win is **smaller recalibration**, not zero. Frame success as
  "shrinks the drift gap / the calibration trials needed", not "solves it".
- Biggest lever is **more data** (more subjects/sessions), not a fancier model.

## Colab setup checklist

1. New notebook -> Runtime -> Change runtime type -> **GPU**.
2. `import torch; torch.cuda.is_available()` must print `True`.
3. Mount Drive, load the uploaded `.npz` arrays.
4. `pip install` only what Colab lacks (PyTorch is pre-installed).
5. Train, checkpoint to Drive, log cross-session accuracy vs the 0.772 bar above.
