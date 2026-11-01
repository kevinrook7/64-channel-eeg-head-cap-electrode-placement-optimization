# Results: personalized channel selection on BCI IV 2a

Motor imagery, left vs right hand. 9 subjects, 2 sessions on different days
(day 1 = T, day 2 = E), 144 trials per session, 22 EEG channels. Chance = 0.500.
All numbers are means across the 9 subjects, taken from the committed runs; this
document recomputes nothing.

## The answer

Two questions, two answers.

1. **Does greedy channel selection beat a dumb baseline?** Yes. At 4 channels,
   greedy picks decode better than both the fixed motor strip and random
   channels, in both time directions, and the edge survives a stronger decoder
   (Filter Bank CSP).
2. **Does a person have a stable "best" channel set?** No. The channels greedy
   picks on day 1 and day 2 are different electrodes, and they do not even cover
   the same scalp region more than random channels would.

So greedy finds a *good* channel set each day, but a *different* good set each
day. There is no single personal fingerprint of best channels here.

## Headline numbers

Cross-session (choose channels + train on one day, score once on the other day):

| test | channels | greedy | motor strip | random | all 22 |
| --- | --- | --- | --- | --- | --- |
| day 1 -> day 2 (plain CSP) | 4 | 0.725 | 0.698 | 0.663 | 0.741 |
| day 2 -> day 1 (plain CSP) | 4 | 0.705 | 0.683 | 0.647 | 0.735 |
| day 1 -> day 2 (FBCSP) | 4 | 0.761 | 0.692 | 0.676 | 0.772 |

Read it as three facts:

- **Greedy > random, both directions.** +0.062 (day1->day2) and +0.059
  (day2->day1) at 4 channels. The two runs agree, so the win is not a lucky
  split.
- **Greedy > motor strip, and the gap grows with a better decoder.** Plain CSP:
  greedy - motor = +0.027 (7/9 subjects). FBCSP: +0.069 (8/9). A stronger
  baseline did not erase the edge, it widened it, so the edge was not an
  artifact of a weak decoder.
- **4 channels cost little.** All 22 channels score 0.741; the best 4 greedy
  channels score 0.725 (plain) / 0.761 (FBCSP). Most of the signal lives in a
  few channels.

## Within-session (train and test on the same day)

| channels | greedy | motor | random | all 22 |
| --- | --- | --- | --- | --- |
| day 1 (T) | 4ch 0.710 | 0.691 | 0.665 | 0.773 |
| day 2 (E) | 4ch 0.740 | 0.705 | 0.674 | 0.806 |

Greedy beats the motor strip clearly only on day 2 (+0.035, 8/9); on day 1 the
gap is small (+0.019, 4/9). By 8-16 channels every method converges, because
with most channels picked they all capture the same signal. Baseline decoding
is healthy: 7 of 9 subjects decode well (vs only 3 of 10 on PhysioNet).

## How few channels is enough

The practical question for a wearable headset is not "do few channels beat
random" but "how few can we keep and still match the full cap". The
cross-session greedy curve (FBCSP, train day 1 -> test day 2, means over 9
subjects) answers it:

| channels | greedy accuracy | vs full 22 |
| --- | --- | --- |
| 4 | 0.761 | -0.011 |
| 8 | 0.777 | +0.005 |
| 16 | 0.796 | +0.024 |
| 22 (all) | 0.772 | — |

Read as a diminishing return:

- **8 channels already match the full 22** (0.777 vs 0.772). Even 4 is within
  ~0.01. Most of the decodable signal lives in a handful of channels.
- **16 slightly beats 22** — dropping the 6 least useful channels removes noise.
- Beyond 8, extra electrodes buy almost nothing on this task.

So the fewest that still works, on this data, is about **8 well-placed
channels**: full-cap accuracy, few enough to be wearable, with enough overlap to
give the decoder slack when the best spot drifts. 4 channels is usable but thin —
little redundancy, so it leans entirely on per-session recalibration.

Two limits on this number: it holds only *with* per-session channel re-selection
(a fixed 4 or 8 degrades, per the stability finding), and "works" here means
left-vs-right hand only — more mental commands would likely need more channels.

## The stability finding

Greedy run separately on each day picks **different** channels; the overlap of
the two k-channel sets is no better than two random sets would overlap.

The region test asked the softer question: even if the electrodes differ, do the
picks land in the same scalp area (neighbouring electrodes carry near-identical
signal)? They do not, by much:

| k | day-2 pick -> nearest day-1 pick | random channels | closer than random | p < 0.05 |
| --- | --- | --- | --- | --- |
| 4 | 3.6 cm | 4.5 cm | 7/9 | 2/9 |
| 8 | 2.6 cm | 2.7 cm | 5/9 | 1/9 |
| 16 | 0.9 cm | 1.0 cm | 6/9 | 0/9 |

Picks are a hair closer than random but almost never pass significance. Only
subjects 1 and 9 cluster clearly. So the day-to-day instability is **geometric**
— not just greedy choosing among correlated neighbours. A better decoder
(FBCSP) does not fix this; it is a property of the data, not the classifier.

## Caveats

- 9 subjects with many comparisons: results are **tentative**, not confirmed.
- Subjects 2 and 5 stay near chance in every regime — they just do not decode
  well; they drag the means down and inflate the spread.
- Selecting a channel does not prove the signal came from the cortex beneath it
  (volume conduction). Channel selection is not source localization.
- No leakage anywhere: channel/band selection and all fitting use training data
  only; nested CV within-session; each cross-session test session is scored once.
  The region test scores no accuracy at all.

## What this means for the project goal

The Level-1 question ("can personalized channel selection decode with fewer
channels than a fixed baseline?") has a measured, honest answer on real
multi-session EEG: **yes on accuracy** (a few good channels rival all 22, and
beat a fixed motor strip), but **a person has no stable best-channel set day to
day**.

For a wearable headset this translates to a concrete design:

- **Count: ~8 electrodes.** That matches the full cap here while staying few
  enough to wear daily and leaving redundancy for drift.
- **Placement: a good *region*, not a pinpoint.** The best exact spot drifts, so
  electrodes are fixed (no moving parts) over the informative motor area with
  enough overlap that a neighbour catches the signal when the best spot shifts.
- **Drift handled in software, by the decoder**, not by hardware: the montage is
  fixed, and the decoder re-trains (and, where the hardware allows, re-selects)
  each session. A fixed-forever per-person montage is the one option the data
  rules out.

The improvement over existing 64-channel rigs is not accuracy — it is
*wearability*: usable accuracy from a cap a person will actually put on every
day, at the cost of a short per-session recalibration.

Parked (need PyTorch, more data, and ideally a GPU): FBCNet, GANs, and other
deep models. Their aim is the *next* level — a decoder that tolerates the drift
well enough to shrink or remove that recalibration. With only 9 subjects they
may not beat CSP; that negative result would itself be informative.
