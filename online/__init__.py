"""Part 2: causal online decoding and quick recalibration on BCI IV 2a.

This package is separate from the Part 1 scripts on purpose. Part 1
(`select_channels.py`, `fbcsp.py`, `bnci_*.py`) produced the frozen v1 result
tagged `v1-channel-selection` and must not change, so nothing here edits those
files or their numbers.

The research question Part 2 answers:

    v1 showed 8 selected channels match the 22-channel accuracy, but the best
    channels drift across sessions. So: can a small, fixed channel set chosen on
    session 1 be *recalibrated quickly* on session 2 and still decode online?

Honest-evaluation contract (every module here obeys it):

  * Channels are selected ONLY on session 1 (the training session). Session 2 is
    the stream and never informs selection -- no test-session leakage.
  * Decoding is CAUSAL: a prediction at time t uses only samples at or before t.
    A sliding window looks backward only; future samples are never read.
  * Accuracy is reported as a function of decision latency (time since cue) and
    of the number of recalibration trials, not as one cue-locked number.
  * The transfer baseline (Euclidean alignment) uses only the UNLABELED incoming
    covariances of the stream -- it never sees session-2 labels.
  * Two classes only. BCI IV 2a has four (left hand, right hand, feet, tongue);
    this uses the left/right subset, as the loader (`load_bnci.LEFT_RIGHT`) already
    restricts to, and that is stated wherever a number is reported.
"""
