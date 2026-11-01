r"""Export BCI IV 2a as .npz arrays for the deep-model work on Colab.

Why this exists: the deep-model step (see docs/deep-model-plan.md) runs on
Google Colab's GPU, which this laptop lacks. Colab has neither MNE nor the
dataset, so we export the SAME preprocessed arrays this repo uses -- here, once
-- and upload the .npz files to Google Drive. Colab then loads plain numpy with
no MNE install and no re-download.

Each .npz holds one subject-session:
    X : float array (trials, channels, times)  -- 8-30 Hz, -0.5..4.0 s around cue
    y : int array   (trials,)                   -- 2 = left hand, 3 = right hand
    ch_names : the EEG channel names (same set for that subject's T and E)
    sfreq    : sampling rate (Hz)

The arrays come from exactly prepare_bnci_sessions() + task_window(), so the
deep model sees the identical signal CSP/FBCSP saw -- the comparison stays fair.

Run (this machine, no GPU needed):
    .\.venv\Scripts\python.exe export_for_colab.py
Output: export_npz/sub{N}_{T,E}.npz for all 9 subjects x 2 sessions (18 files).
Upload the export_npz/ folder to Drive, e.g. MyDrive/myeeg_data/.
"""

import argparse
from pathlib import Path

import mne
import numpy as np

from baseline import task_window
from bnci_evaluate import SUBJECTS
from load_bnci import SESSIONS, prepare_bnci_sessions

OUT_DIR = Path(__file__).resolve().parent / "export_npz"


def export_subject(subject, out_dir):
    """Write sub{subject}_T.npz and sub{subject}_E.npz; return (shape_T, shape_E)."""
    sessions = prepare_bnci_sessions(subject)  # {"T": epochs, "E": epochs}, shared channels
    shapes = {}
    for session in SESSIONS:  # "T", "E"
        epochs = sessions[session]
        X = task_window(epochs).astype(np.float32)  # (trials, channels, times)
        y = epochs.events[:, 2].astype(np.int64)    # 2 = left, 3 = right
        path = out_dir / f"sub{subject}_{session}.npz"
        np.savez_compressed(
            path,
            X=X,
            y=y,
            ch_names=np.array(epochs.ch_names),
            sfreq=np.float64(epochs.info["sfreq"]),
        )
        shapes[session] = X.shape
        print(f"  {path.name}: X {X.shape}  y {y.shape}  "
              f"({(y == 2).sum()} left / {(y == 3).sum()} right)")
    return shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS),
                        help="subject numbers to export (default: all 9)")
    parser.add_argument("--out", type=Path, default=OUT_DIR,
                        help="output directory for the .npz files")
    args = parser.parse_args()

    mne.set_log_level("WARNING")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(args.subjects)} subject(s) to {args.out}")

    for subject in args.subjects:
        print(f"subject {subject}:")
        export_subject(subject, args.out)

    files = sorted(args.out.glob("sub*_*.npz"))
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\nDone. {len(files)} files, {total_mb:.1f} MB total in {args.out}")
    print("Upload this folder to Google Drive (e.g. MyDrive/myeeg_data/) for Colab.")


if __name__ == "__main__":
    main()
