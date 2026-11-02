"""Deep-model cross-session experiment for BCI IV 2a (Colab/GPU).

Runs the three deep-decoder variants compared against FBCSP in
docs/results-summary.md, all on the same data exported by export_for_colab.py:

    1. EEGNet, trained per subject          (mean cross-session acc 0.721)
    2. EEGNet + domain adaptation (DANN)    (0.718)
    3. EEGNet, pooled over all 9 subjects   (0.728)

Baseline to beat: FBCSP cross-session = 0.772. All three lose to it -- deep
learning did not beat CSP for few-channel cross-session motor imagery on this
dataset (9 subjects). See docs/results-summary.md for the full write-up and
docs/deep-model-plan.md for the plan.

Data: each sub{N}_{T,E}.npz holds
    X        (trials, 22, 876) float, 8-30 Hz, task window
    y        (trials,)         labels already 0 = left, 1 = right
    ch_names (22,)             channel names
    sfreq                      250.0

Usage (Colab): set DATA_DIR to the Drive folder holding the .npz files, pick a
GPU runtime, then run. Locally: point DATA_DIR at ./export_npz (CPU works but is
slow). Written to run top-to-bottom; the three experiments are independent.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

# Colab: '/content/drive/MyDrive/MyEEG_data'   Local: 'export_npz'
DATA_DIR = "export_npz"
EPOCHS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_session(subject, session):
    """Load one subject-session .npz -> (X float32, y int64 in {0, 1})."""
    d = np.load(f"{DATA_DIR}/sub{subject}_{session}.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def norm(X):
    """Per-trial, per-channel z-score over time. Mild defence against drift."""
    m = X.mean(axis=2, keepdims=True)
    s = X.std(axis=2, keepdims=True) + 1e-7
    return (X - m) / s


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class EEGNet(nn.Module):
    """Compact EEGNet-style net: temporal conv -> depthwise spatial -> separable."""

    def __init__(self, n_ch=22, n_time=876, n_classes=2):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, (n_ch, 1), groups=16, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
        )
        self.sep = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.5),
        )
        with torch.no_grad():
            d = self.sep(self.spatial(self.temporal(torch.zeros(1, 1, n_ch, n_time))))
            flat = d.flatten(1).shape[1]
        self.head = nn.Linear(flat, n_classes)

    def forward(self, x):            # x: (batch, n_ch, n_time)
        x = x.unsqueeze(1)
        x = self.sep(self.spatial(self.temporal(x)))
        return self.head(x.flatten(1))


class GradReverse(Function):
    """Identity forward, sign-flipped gradient backward (DANN)."""

    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb=1.0):
    return GradReverse.apply(x, lamb)


class DANN(nn.Module):
    """Shared features + class head + domain head through gradient reversal."""

    def __init__(self, n_ch=22, n_time=876):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (n_ch, 1), groups=16, bias=False), nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False), nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.5),
        )
        with torch.no_grad():
            flat = self.feat(torch.zeros(1, 1, n_ch, n_time)).flatten(1).shape[1]
        self.cls = nn.Linear(flat, 2)   # left vs right
        self.dom = nn.Linear(flat, 2)   # day1 vs day2

    def forward(self, x, lamb=0.0):
        f = self.feat(x.unsqueeze(1)).flatten(1)
        return self.cls(f), self.dom(grad_reverse(f, lamb))


# --------------------------------------------------------------------------- #
# experiment 1: EEGNet per subject
# --------------------------------------------------------------------------- #
def train_eegnet(Xtr, ytr, Xte, yte, epochs=EPOCHS):
    Xtr, Xte = norm(Xtr), norm(Xte)
    Xtr = torch.tensor(Xtr).to(device); ytr = torch.tensor(ytr).to(device)
    Xte = torch.tensor(Xte).to(device); yte = torch.tensor(yte).to(device)

    model = EEGNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()


def run_per_subject():
    accs = []
    for s in range(1, 10):
        Xtr, ytr = load_session(s, "T")
        Xte, yte = load_session(s, "E")
        a = train_eegnet(Xtr, ytr, Xte, yte)
        accs.append(a)
        print(f"  subject {s}: day2 acc = {a:.3f}")
    print(f"  MEAN per-subject EEGNet: {np.mean(accs):.3f}")
    return np.mean(accs)


# --------------------------------------------------------------------------- #
# experiment 2: EEGNet + domain adaptation (DANN)
# --------------------------------------------------------------------------- #
def train_dann(Xtr, ytr, Xte, yte, epochs=EPOCHS):
    Xtr, Xte = norm(Xtr), norm(Xte)
    Xtr = torch.tensor(Xtr).to(device); ytr = torch.tensor(ytr).to(device)
    Xte = torch.tensor(Xte).to(device); yte = torch.tensor(yte).to(device)

    model = DANN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        lamb = 2.0 / (1 + np.exp(-10 * ep / epochs)) - 1   # ramp 0 -> 1
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]
            xb, yb = Xtr[idx], ytr[idx]
            tgt = torch.randint(0, len(Xte), (len(idx),), device=device)
            xt = Xte[tgt]                                   # unlabeled day-2 batch

            opt.zero_grad()
            cls_s, dom_s = model(xb, lamb)
            _, dom_t = model(xt, lamb)
            d_s = torch.zeros(len(idx), dtype=torch.long, device=device)   # day1 = 0
            d_t = torch.ones(len(idx), dtype=torch.long, device=device)    # day2 = 1
            loss = lossf(cls_s, yb) + lossf(dom_s, d_s) + lossf(dom_t, d_t)
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pred, _ = model(Xte, 0.0)
        return (pred.argmax(1) == yte).float().mean().item()


def run_dann():
    accs = []
    for s in range(1, 10):
        Xtr, ytr = load_session(s, "T")
        Xte, yte = load_session(s, "E")
        a = train_dann(Xtr, ytr, Xte, yte)
        accs.append(a)
        print(f"  subject {s}: day2 acc = {a:.3f}")
    print(f"  MEAN EEGNet + DANN: {np.mean(accs):.3f}")
    return np.mean(accs)


# --------------------------------------------------------------------------- #
# experiment 3: EEGNet pooled over all 9 subjects
# --------------------------------------------------------------------------- #
def run_pooled(epochs=EPOCHS):
    Xtr_list, ytr_list, test_sets = [], [], {}
    for s in range(1, 10):
        Xt, yt = load_session(s, "T")
        Xe, ye = load_session(s, "E")
        Xtr_list.append(norm(Xt)); ytr_list.append(yt)
        test_sets[s] = (norm(Xe), ye)

    Xtr = torch.tensor(np.concatenate(Xtr_list)).to(device)
    ytr = torch.tensor(np.concatenate(ytr_list)).to(device)
    print(f"  pooled train: {tuple(Xtr.shape)}")

    model = EEGNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()

    model.eval()
    accs = []
    with torch.no_grad():
        for s in range(1, 10):
            Xe, ye = test_sets[s]
            Xe = torch.tensor(Xe).to(device); ye = torch.tensor(ye).to(device)
            a = (model(Xe).argmax(1) == ye).float().mean().item()
            accs.append(a)
            print(f"  subject {s}: day2 acc = {a:.3f}")
    print(f"  MEAN pooled EEGNet: {np.mean(accs):.3f}")
    return np.mean(accs)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(f"device: {device}\n")
    print("[1] EEGNet per subject (expect ~0.721)")
    m1 = run_per_subject()
    print("\n[2] EEGNet + domain adaptation (expect ~0.718)")
    m2 = run_dann()
    print("\n[3] EEGNet pooled over 9 subjects (expect ~0.728)")
    m3 = run_pooled()
    print("\n--- summary (cross-session, train day1 -> test day2) ---")
    print(f"  FBCSP baseline        0.772")
    print(f"  EEGNet per subject    {m1:.3f}")
    print(f"  EEGNet + DANN         {m2:.3f}")
    print(f"  EEGNet pooled         {m3:.3f}")
    print("  verdict: all deep variants lose to CSP on this dataset.")
