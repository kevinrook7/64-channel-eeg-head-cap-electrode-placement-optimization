No, copy-pasting the entire block at once is not the best way to learn.

While it will instantly build a working app, it turns you into a passive spectator. You might see files appear in your workspace, but you won't actually absorb how the neural data maps to the code or why the machine learning loop makes its choices.

Instead, the best way to learn BCI and ML development is Socratic, iterative building—where you write code piece by piece, treat the AI as a pair-programmer, and intentionally break things to see how they work.

Here is an alternative, high-retention framework you can use inside Cursor to actually master the material:

> **Direction update (2026-10-09):** The project is now about **personalized EEG channel selection for motor-imagery BCIs** (see [`AGENTS.md`](../AGENTS.md)). The phases below were reframed for that goal. The original "focus score" and contextual-bandit phases are kept as optional extensions at the end.

------------------------------

## The "Build-and-Deconstruct" Learning Blueprint

Instead of giving Cursor a massive blueprint to automate the whole project, feed it one concept at a time using these sequential steps. Finish each phase, and run its small experiment, before starting the next one.

## 📂 Phase 1: Meet Real EEG (Dataset Loading)

Create `load_data.py` and prompt Cursor with:

> "I want to load one subject from the PhysioNet EEG Motor Movement/Imagery dataset with MNE. Don't write the whole thing for me. Explain what an EDF file is, what runs 4, 8, and 12 contain, and what the T0/T1/T2 annotations mean. Then give me just the code to load one run and print its channel names, sampling rate, and events."

* **How this helps you learn:** You see real microvolt recordings, and you learn that labels depend on the run number. Check yourself: why can't you treat T1 as "left hand" in every run?

## 🧹 Phase 2: Clean and Validate the Signal (Preprocessing + Channels)

Create `preprocess.py` and prompt:

> "Explain band-pass filtering and why motor imagery uses roughly 8–30 Hz (mu and beta). Show me how to filter, cut the data into epochs around T1/T2, standardize channel names, and attach a 10-10 montage. Help me write a check that flags flat or very noisy channels."

* **How this helps you learn:** You learn that "64 channels" doesn't automatically mean "64 usable channels." **Break it:** filter at 1–4 Hz instead and look at how the epochs change.

## 📏 Phase 3: Build the Baseline (All Channels)

Create `baseline.py` and prompt:

> "Before any optimization, I want a reproducible all-channel baseline for left vs right imagery for one subject. Explain log band-power features (and later CSP), then logistic regression or LDA. Explain cross-validation and why chance level is 50%. Fix the random seeds."

* **How this helps you learn:** You get the number every later experiment has to beat. Ask yourself: what accuracy would a classifier that always guesses "left" get?

## 🎯 Phase 4: Select Channels Without Cheating (Optimization + Leakage)

Create `select_channels.py` and prompt:

> "Now I want to choose 8, 16, and 32 channel subsets. Start with greedy forward selection. Before you write code, ask me why channel selection must happen inside the training folds only. Then show me nested cross-validation and an explicit objective like accuracy − λ × number of channels."

* **How this helps you learn:** You learn data leakage, the most common way BCI results get inflated. **Break it:** select channels on all of the data first, then cross-validate, and watch the accuracy look suspiciously good.

## 📊 Phase 5: Compare, Measure Uncertainty, Personalize

Create `evaluate.py` and prompt:

> "Compare full-channel, fixed motor-strip (C3/Cz/C4 area), random subsets, and greedy subsets across several subjects. Show mean ± std, plot accuracy vs channel count, and print which channels each subject got. Explain why picking C3 doesn't prove the signal came from the cortex under C3."

* **How this helps you learn:** You learn to report results honestly, and that the "best" channels can differ from person to person, which is the whole point of personalization.

------------------------------

## 🧩 Optional extensions (only after Phase 5)

* **LSL simulator & real-time streaming** (the original Phase 1, `streamer.py` with `pylsl`): good for testing the plumbing (streams, buffers, latency). Simulated EEG **cannot** show that a decoder works on real brains.
* **Real-time inference:** run a trained subset decoder on a live or replayed LSL stream.
* **"Focus score" from alpha/beta power** (the original Phase 2): alpha/beta power is not a reliable focus or intention measure by itself. Only revisit this with a defined task and labeled data.
* **Contextual-bandit recommendation agent** (the original Phase 3): keep it only if it serves a defined BCI objective, for example choosing which channel configuration to try next.

------------------------------

## 💡 3 Rules for Learning with a Cursor Agent

1. **The "Why" Rule:** Never accept code from Cursor unless you can explain what every single line does. If a block of code looks confusing, highlight it, press Cmd/Ctrl + L (Chat), and ask: "Explain this specific block to me like I am a beginner."
2. **Break it Intentionally:** Once Cursor writes a piece of code that works, change a number. Shrink the channel set from 16 to 4, or move the band-pass from 8–30 Hz to 1–4 Hz. See how accuracy reacts. Observing how a system breaks teaches you exactly how it functions.
3. **Use the Interactive Terminal:** Run your scripts manually in the PowerShell terminal (`python baseline.py`). Watch the print statements scroll by in real-time.

Also keep track of what's **implemented**, what's **tested**, and what's only **planned**, in the status table in [`AGENTS.md`](../AGENTS.md#11-status).
