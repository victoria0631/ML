"""
ML_V9.py - Room occupancy classification (0-3 persons) from 60GHz radar
Based on V6 (best so far, 0.94979 on Kaggle) with several improvements:
- replaced flips/rotations with translation+noise only (makes more sense physically)
- focal loss instead of weighted CE
- kept ConvNeXt-tiny (V7 signal channels and V8 Swin were worse)
- class-3 oversampling done online instead of using the old augmented file
- normalization from our data instead of ImageNet stats
- added 7 handcrafted features (peak position, spread, etc.) concatenated to CNN output
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torch.optim.swa_utils import AveragedModel, SWALR
import torchvision.transforms as T
import timm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---- Reproducibility ----

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---- Config ----

class Config:
    # ── Paths (change these for Colab: just update the four variables below) ──
    TRAIN_DIR     = r"C:\Users\Victoire\Documents\5-persoVictoire\BRNO-VUT\Project_ML\data\x_train"
    TEST_DIR      = r"C:\Users\Victoire\Documents\5-persoVictoire\BRNO-VUT\Project_ML\data\x_test"
    TRAIN_LABELS  = r"C:\Users\Victoire\Documents\5-persoVictoire\BRNO-VUT\Project_ML\data\y_train_v2.csv"
    SUBMISSION_EX = r"C:\Users\Victoire\Documents\5-persoVictoire\BRNO-VUT\Project_ML\data\y_test_submission_example_v2.csv"
    OUT_DIR       = r"C:\Users\Victoire\Documents\5-persoVictoire\BRNO-VUT\Project_ML"

    # ── Model ──────────────────────────────────────────────────────────────────
    MODEL_NAME   = "convnext_tiny"   
    NUM_CLASSES  = 4
    IMG_SIZE     = 128               # native is 51x45, we resize
    N_SIG_FEAT   = 7                 # number of handcrafted features

    # ── Training ───────────────────────────────────────────────────────────────
    BATCH_SIZE    = 48
    EPOCHS_HEAD   = 10               # phase 1 : backbone frozen, head only
    EPOCHS_FULL   = 50               # phase 2 : everything unfrozen
    TOTAL_EPOCHS  = 60

    LR_HEAD       = 1e-3
    LR_BACKBONE   = 5e-5             
    WEIGHT_DECAY  = 1e-4
    GRAD_CLIP     = 1.0

    NUM_FOLDS     = 5
    SWA_START_EP  = 40              
    SWA_LR        = 1e-5
    SWA_ANNEAL    = 5                

    AUG_MULT_CLS3 = 2

    FOCAL_GAMMA   = 2.0

    TTA_ROUNDS    = 5

cfg = Config()
CLASS_NAMES = ["0 person", "1 person", "2 persons", "3 persons"]

# ---- Focal Loss ----

class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy (well-classified) examples so the model
    concentrates gradient on the hard 1↔2 person boundary.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)                          # probability of correct class
        return ((1.0 - pt) ** self.gamma * ce).mean()

# ---- Handcrafted signal features ----

def extract_signal_features(pil_img: Image.Image) -> np.ndarray:
    """
    Compute 7 physics-informed features from the RAW (pre-transform) image.
    Applied on native 51×45 pixels — no resize, no normalisation — so features
    are stable and not affected by augmentation applied to the CNN input.

    Features that discriminate the hard classes:
      1-2: peak_x_norm     — 0-person peak is on the LEFT edge; others centred
      3:   peak_y_norm     — vertical position of brightest point
      4:   spread_x        — horizontal spread of bright region
      5:   spread_y        — vertical spread (wider for multi-person)
      6:   hv_ratio        — row_energy / (row+col) — 1-person is ~horizontal,
                             2-person is more balanced, 3-person is symmetric
      7:   peak_intensity  — normalised max pixel value
      8:   mean_energy     — mean brightness (correlates with person count)
    """
    arr  = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0  # H×W×3
    sig  = arr.max(axis=2)   # H×W  max channel = signal amplitude proxy
    h, w = sig.shape

    # Peak location
    flat = int(sig.argmax())
    py, px = divmod(flat, w)
    px_n = px / max(w - 1, 1)
    py_n = py / max(h - 1, 1)

    # Spread at 80th-percentile threshold
    thresh = float(np.percentile(sig, 80))
    ys, xs = np.where(sig > thresh)
    spread_x = float(xs.std()) / w if len(xs) > 1 else 0.0
    spread_y = float(ys.std()) / h if len(ys) > 1 else 0.0

    # Horizontal vs vertical energy through the peak pixel
    row_e = float(sig[py, :].sum())
    col_e = float(sig[:, px].sum())
    hv_ratio = row_e / (row_e + col_e + 1e-8)   # 1 = pure horizontal

    peak_val  = float(sig[py, px])
    mean_e    = float(sig.mean())

    return np.array([px_n, py_n, spread_x, spread_y,
                     hv_ratio, peak_val, mean_e], dtype=np.float32)

# ---- Dataset normalization ----

def compute_dataset_stats(df: pd.DataFrame, img_dir: str,
                          img_size: int = 128, max_samples: int = 3000):
    """Compute channel mean and std from this dataset (not ImageNet)."""
    print("Computing dataset normalisation statistics …")
    to_t = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    sample = df.sample(min(max_samples, len(df)), random_state=SEED)
    ch_sum = torch.zeros(3)
    ch_sq  = torch.zeros(3)
    n = 0
    for _, row in sample.iterrows():
        t = to_t(Image.open(os.path.join(img_dir, row["filename"])).convert("RGB"))
        ch_sum += t.mean([1, 2])
        ch_sq  += (t ** 2).mean([1, 2])
        n += 1
    mean = ch_sum / n
    std  = ((ch_sq / n) - mean ** 2).clamp(min=1e-8).sqrt()
    print(f"  mean = {[round(v, 4) for v in mean.tolist()]}")
    print(f"  std  = {[round(v, 4) for v in std.tolist()]}")
    return mean.tolist(), std.tolist()

# ---- Transforms ----

class AddGaussianNoise:
    """
    Additive Gaussian noise simulating realistic SNR variation in the radar.
    Applied AFTER ToTensor(), before Normalize().
    sigma=0.012 matches the background noise floor visible in the raw images.
    """
    def __init__(self, sigma: float = 0.012):
        self.sigma = sigma

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return (t + torch.randn_like(t) * self.sigma).clamp(0.0, 1.0)


def build_transform(mean, std, mode: str = "train") -> T.Compose:
    """
    mode='train' — translation + noise (Fix 1: NO flips/rotations/colorjitter)
    mode='val'   — deterministic; no augmentation
    mode='tta'   — same as train but lighter noise (for test-time inference)
    """
    ops = [T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE))]
    if mode in ("train", "tta"):
        ops.append(T.RandomAffine(degrees=0, translate=(0.04, 0.04)))
    ops.append(T.ToTensor())
    if mode in ("train", "tta"):
        sigma = 0.012 if mode == "train" else 0.007
        ops.append(AddGaussianNoise(sigma=sigma))
    ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)

#DATASETS


class RadarDataset(Dataset):
    """
    Returns (img_tensor, signal_features, label_or_id).
    Signal features are always extracted from the original PIL image
    (before any spatial augmentation) so they are stable and deterministic.
    """
    def __init__(self, df: pd.DataFrame, img_dir: str,
                 transform: T.Compose, is_test: bool = False):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = img_dir
        self.transform = transform
        self.is_test  = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        pil  = Image.open(os.path.join(self.img_dir, row["filename"])).convert("RGB")
        sig  = torch.from_numpy(extract_signal_features(pil))   # (7,)
        img  = self.transform(pil)                               # (3, H, W)
        if self.is_test:
            return img, sig, int(row["id"])
        return img, sig, int(row["target"])


class Class3AugDataset(Dataset):
    """
    Fix 4 — Generates (AUG_MULT_CLS3 - 1) physically valid augmented copies
    of every class-3 training sample, online, without touching existing files.

    Why class-3 only: class 0 is always 100% accurate; class 3 is the
    most underrepresented class (977 samples vs 3786 for class 1).
    """
    def __init__(self, df_c3: pd.DataFrame, img_dir: str, mean, std):
        self.df      = df_c3.reset_index(drop=True)
        self.img_dir = img_dir
        self.aug     = build_transform(mean, std, mode="train")
        self.extra   = cfg.AUG_MULT_CLS3 - 1   # extra copies per real sample

    def __len__(self):
        return len(self.df) * self.extra

    def __getitem__(self, idx):
        row = self.df.iloc[idx % len(self.df)]
        pil = Image.open(os.path.join(self.img_dir, row["filename"])).convert("RGB")
        sig = torch.from_numpy(extract_signal_features(pil))
        return self.aug(pil), sig, 3

# ---- Model ----

class OccupancyV9(nn.Module):
    """
    ConvNeXt-tiny backbone (proven best in V6) with a small signal-feature MLP
    whose output is concatenated to the CNN embedding before the classifier.

    Backbone (768-d) ──┐
                        cat(800-d) → LayerNorm → Dropout → Linear(256)
    SigMLP (7→32-d) ───┘              → GELU → LayerNorm → Dropout → Linear(4)
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=0)
        cnn_dim = self.backbone.num_features   # 768 for convnext_tiny

        # Lightweight MLP for handcrafted signal features
        self.sig_proj = nn.Sequential(
            nn.Linear(cfg.N_SIG_FEAT, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Classifier head over concatenated features
        self.head = nn.Sequential(
            nn.LayerNorm(cnn_dim + 32),
            nn.Dropout(0.30),
            nn.Linear(cnn_dim + 32, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.20),
            nn.Linear(256, cfg.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.backbone(x)          # (B, 768)
        sig_feat = self.sig_proj(sig)        # (B, 32)
        return self.head(torch.cat([cnn_feat, sig_feat], dim=1))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

# ---- Training helpers ----

def run_epoch(model, loader, criterion, optimizer=None):
    """
    Single epoch — train if optimizer is given, eval otherwise.
    Returns (loss, accuracy, predictions, labels).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, sigs, targets in loader:
            imgs    = imgs.to(DEVICE)
            sigs    = sigs.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(imgs, sigs)
            loss   = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

            preds       = logits.argmax(1)
            total_loss += loss.item() * targets.size(0)
            correct    += preds.eq(targets).sum().item()
            total      += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    return total_loss / total, 100.0 * correct / total, \
           np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def bn_update(swa_model: AveragedModel, loader):
    """
    Reset and recompute BatchNorm running stats for the SWA model.
    ConvNeXt uses LayerNorm so this is effectively a no-op here, but kept
    for correctness in case any BN layers are present (e.g. in sig_proj).
    Uses our 3-tuple (img, sig, label) loader format.
    """
    swa_model.train()
    for m in swa_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None   # cumulative moving average
    for imgs, sigs, _ in loader:
        swa_model(imgs.to(DEVICE), sigs.to(DEVICE))

# ---- Fold training ----

def train_fold(df: pd.DataFrame, fold_idx: int, mean, std):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1} / {cfg.NUM_FOLDS}")
    print(f"{'='*60}")

    # Split 
    skf = StratifiedKFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = list(skf.split(df, df["target"]))[fold_idx]
    df_tr = df.iloc[tr_idx]
    df_va = df.iloc[va_idx]
    print(f"  Train: {len(df_tr)}  |  Val: {len(df_va)}")

    # Datasets 
    tr_tf  = build_transform(mean, std, "train")
    va_tf  = build_transform(mean, std, "val")

    base_ds = RadarDataset(df_tr, cfg.TRAIN_DIR, tr_tf)
    aug3_ds = Class3AugDataset(df_tr[df_tr["target"] == 3], cfg.TRAIN_DIR, mean, std)
    tr_ds   = ConcatDataset([base_ds, aug3_ds])
    va_ds   = RadarDataset(df_va, cfg.TRAIN_DIR, va_tf)

    n_aug3 = len(aug3_ds)
    print(f"  Class-3 augmented extras: {n_aug3}  "
          f"(total training samples: {len(tr_ds)})")

    # weighted sampler to balance classes
    lbl_all = list(df_tr["target"].values) + [3] * n_aug3
    counts  = np.bincount(lbl_all, minlength=cfg.NUM_CLASSES).astype(float)
    smp_w   = torch.tensor([1.0 / counts[l] for l in lbl_all])
    sampler = WeightedRandomSampler(smp_w, len(smp_w), replacement=True)

    kw = dict(num_workers=0, pin_memory=True)
    tr_loader = DataLoader(tr_ds, cfg.BATCH_SIZE, sampler=sampler, **kw)
    va_loader = DataLoader(va_ds, cfg.BATCH_SIZE, shuffle=False, **kw)

    # model + focal loss with class weights
    model = OccupancyV9().to(DEVICE)

    # Fix 2: Focal loss with inverse-frequency class weights
    cw        = torch.tensor(1.0 / counts, dtype=torch.float32)
    cw        = (cw / cw.sum() * cfg.NUM_CLASSES).to(DEVICE)
    criterion = FocalLoss(gamma=cfg.FOCAL_GAMMA, weight=cw)

    # ── History bookkeeping ────────────────────────────────────────────────────
    history = {"tl": [], "vl": [], "ta": [], "va": []}
    best_va      = 0.0
    best_state   = None
    best_vp      = None
    best_vlbl    = None

    # --- Phase 1: frozen backbone, train head only ---

    model.freeze_backbone()
    opt1 = optim.AdamW(
        list(model.sig_proj.parameters()) + list(model.head.parameters()),
        lr=cfg.LR_HEAD, weight_decay=cfg.WEIGHT_DECAY,
    )
    sch1 = optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=cfg.EPOCHS_HEAD, eta_min=1e-5
    )

    for ep in range(cfg.EPOCHS_HEAD):
        tl, ta, _,  _   = run_epoch(model, tr_loader, criterion, opt1)
        vl, va, vp, vlbl = run_epoch(model, va_loader, criterion)
        sch1.step()
        for k, v in zip(("tl","vl","ta","va"), (tl, vl, ta, va)):
            history[k].append(v)
        if va > best_va:
            best_va    = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_vp, best_vlbl = vp, vlbl
        print(f"  [Head  {ep+1:2d}/{cfg.EPOCHS_HEAD}]  "
              f"Train {ta:.1f}%  Val {va:.1f}%  Best {best_va:.1f}%")

    
    # --- Phase 2: unfreeze everything + SWA ---
    model.unfreeze_backbone()
    opt2 = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.LR_BACKBONE},
        {"params": model.sig_proj.parameters(), "lr": cfg.LR_HEAD * 0.3},
        {"params": model.head.parameters(),     "lr": cfg.LR_HEAD * 0.3},
    ], weight_decay=cfg.WEIGHT_DECAY)

    sch2     = optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=cfg.EPOCHS_FULL, eta_min=cfg.SWA_LR / 2
    )
    swa_model  = AveragedModel(model)
    swa_sch    = SWALR(opt2, swa_lr=cfg.SWA_LR, anneal_epochs=cfg.SWA_ANNEAL)
    swa_active = False

    for ep in range(cfg.EPOCHS_FULL):
        abs_ep = ep + cfg.EPOCHS_HEAD
        tl, ta, _,   _    = run_epoch(model, tr_loader, criterion, opt2)
        vl, va, vp,  vlbl = run_epoch(model, va_loader, criterion)

        if abs_ep >= cfg.SWA_START_EP:
            swa_model.update_parameters(model)
            swa_sch.step()
            swa_active = True
        else:
            sch2.step()

        for k, v in zip(("tl","vl","ta","va"), (tl, vl, ta, va)):
            history[k].append(v)
        if va > best_va:
            best_va    = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_vp, best_vlbl = vp, vlbl

        if (ep + 1) % 5 == 0:
            tag = "  [SWA]" if swa_active else ""
            print(f"  [Full {ep+1:3d}/{cfg.EPOCHS_FULL}]  "
                  f"Train {ta:.1f}%  Val {va:.1f}%  Best {best_va:.1f}%{tag}")

    if swa_active:
        bn_update(swa_model, tr_loader)

    # save best checkpoint

    model.load_state_dict(best_state)
    print(f"\n  Best val acc: {best_va:.2f}%")

    torch.save({"state_dict": best_state, "val_acc": best_va},
               os.path.join(cfg.OUT_DIR, f"v9_fold{fold_idx}.pth"))
    if swa_active:
        torch.save(swa_model.state_dict(),
                   os.path.join(cfg.OUT_DIR, f"v9_swa_fold{fold_idx}.pth"))

    return model, swa_model if swa_active else model, history, best_va, best_vp, best_vlbl

# ---- Ensemble + TTA inference ----

@torch.no_grad()
def predict_tta(models: list, df_test: pd.DataFrame, mean, std) -> tuple:
    """
    Average softmax probabilities across all models and TTA passes.
    TTA uses only valid augmentations (Fix 1: translation + noise).
    Signal features are deterministic (computed from original image, no TTA).
    """
    n_models = len(models)
    print(f"\nPredicting: {n_models} model(s) × {cfg.TTA_ROUNDS + 1} passes …")

    val_tf = build_transform(mean, std, "val")
    tta_tf = build_transform(mean, std, "tta")
    kw     = dict(batch_size=cfg.BATCH_SIZE, shuffle=False,
                  num_workers=0, pin_memory=True)

    all_probs = []

    for model in models:
        model.eval()

        # Standard pass (no augmentation)
        ds = RadarDataset(df_test, cfg.TEST_DIR, val_tf, is_test=True)
        probs = []
        for imgs, sigs, _ in DataLoader(ds, **kw):
            out = model(imgs.to(DEVICE), sigs.to(DEVICE))
            probs.append(F.softmax(out, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs))

        # TTA passes — light augmentation, same signal features
        for _ in range(cfg.TTA_ROUNDS):
            ds_tta = RadarDataset(df_test, cfg.TEST_DIR, tta_tf, is_test=True)
            probs  = []
            for imgs, sigs, _ in DataLoader(ds_tta, **kw):
                out = model(imgs.to(DEVICE), sigs.to(DEVICE))
                probs.append(F.softmax(out, dim=1).cpu().numpy())
            all_probs.append(np.concatenate(probs))

    avg   = np.mean(all_probs, axis=0)   # (N_test, 4)
    preds = avg.argmax(axis=1)
    print(f"  Distribution: {dict(sorted(Counter(preds.tolist()).items()))}")
    return preds, avg

# ---- Main ----

if __name__ == "__main__":

    # ── Load labels ────────────────────────────────────────────────────────────
    df_train = pd.read_csv(cfg.TRAIN_LABELS)
    df_sub   = pd.read_csv(cfg.SUBMISSION_EX)
    df_train["filename"] = df_train["id"].apply(lambda x: f"img_{x+1}.png")
    df_sub["filename"]   = df_sub["id"].apply(lambda x: f"img_{x+1}.png")

    print(f"Train: {len(df_train)}  |  Test: {len(df_sub)}")
    print("Class distribution:", df_train["target"].value_counts().sort_index().to_dict())

        # compute normalization from our data

    MEAN, STD = compute_dataset_stats(df_train, cfg.TRAIN_DIR)

    fold_models  = []    # best-checkpoint models
    fold_swa     = []    # SWA-averaged models
    fold_accs    = []
    all_histories = []
    oof_preds, oof_labels = [], []

    for fold in range(cfg.NUM_FOLDS):
        model, swa_m, hist, vacc, vp, vlbl = train_fold(df_train, fold, MEAN, STD)
        fold_models.append(model)
        fold_swa.append(swa_m)
        fold_accs.append(vacc)
        all_histories.append(hist)
        if vp is not None:
            oof_preds.extend(vp.tolist())
            oof_labels.extend(vlbl.tolist())

    print(f"\n{'='*60}")
    print("  FOLD SUMMARY")
    print(f"{'='*60}")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1}: {acc:.2f}%")
    print(f"  Mean : {np.mean(fold_accs):.2f}%  ±  {np.std(fold_accs):.2f}%")
    print(f"  Best : {max(fold_accs):.2f}%")

    # ── Training curves ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"V9 Training — Mean {np.mean(fold_accs):.2f}%  |  Best {max(fold_accs):.2f}%",
        fontsize=13
    )
    for h in all_histories:
        axes[0].plot(h["tl"], alpha=0.35, color="steelblue")
        axes[0].plot(h["vl"], alpha=0.35, color="tomato")
        axes[1].plot(h["ta"], alpha=0.35, color="steelblue")
        axes[1].plot(h["va"], alpha=0.35, color="tomato")
    for ax in axes:
        ax.axvline(cfg.EPOCHS_HEAD,    color="grey",  ls="--", alpha=0.6, label="Unfreeze")
        ax.axvline(cfg.SWA_START_EP,   color="green", ls="--", alpha=0.6, label="SWA start")
        ax.legend(["Train", "Val", "Unfreeze", "SWA"])
    axes[0].set(xlabel="Epoch", ylabel="Loss",         title="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, "v9_training.png"), dpi=150)
    plt.show()
    print("Saved: v9_training.png")

    # OOF confusion matrix 
    if oof_preds:
        oof_acc = accuracy_score(oof_labels, oof_preds) * 100
        cm = confusion_matrix(oof_labels, oof_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set(title=f"V9 OOF Confusion Matrix — {oof_acc:.2f}%",
               xlabel="Predicted", ylabel="True")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUT_DIR, "v9_confusion.png"), dpi=150)
        plt.show()
        print(f"Saved: v9_confusion.png  (OOF accuracy: {oof_acc:.2f}%)")
        print(classification_report(oof_labels, oof_preds, target_names=CLASS_NAMES))

    # generate submissions 
    # 1) Full 5-fold ensemble 
    preds_ens, _ = predict_tta(fold_models, df_sub, MEAN, STD)
    sub_ens = df_sub[["id"]].copy()
    sub_ens["target"] = preds_ens
    sub_ens.to_csv(os.path.join(cfg.OUT_DIR, "sub_v9_ensemble.csv"), index=False)
    print("\nSaved: sub_v9_ensemble.csv")

    # 2) Best single-fold model
    best_fold = int(np.argmax(fold_accs))
    preds_best, _ = predict_tta([fold_models[best_fold]], df_sub, MEAN, STD)
    sub_best = df_sub[["id"]].copy()
    sub_best["target"] = preds_best
    sub_best.to_csv(os.path.join(cfg.OUT_DIR, "sub_v9_best.csv"), index=False)
    print(f"Saved: sub_v9_best.csv  (fold {best_fold+1}, val={fold_accs[best_fold]:.2f}%)")

    # 3) SWA ensemble
    preds_swa, _ = predict_tta(fold_swa, df_sub, MEAN, STD)
    sub_swa = df_sub[["id"]].copy()
    sub_swa["target"] = preds_swa
    sub_swa.to_csv(os.path.join(cfg.OUT_DIR, "sub_v9_swa.csv"), index=False)
    print("Saved: sub_v9_swa.csv")

    print(f"""
{'='*60}
  V9 COMPLETE
{'='*60}
  OOF mean accuracy : {np.mean(fold_accs):.2f}%
  Best fold          : {max(fold_accs):.2f}%

  Submit in this order (best first):
    1. sub_v9_swa.csv        ← SWA ensemble, usually most stable
    2. sub_v9_ensemble.csv   ← best-checkpoint ensemble
    3. sub_v9_best.csv       ← single best fold (baseline check)

  V6 reference (Kaggle): ensemble=0.94979, best=0.93711
{'='*60}
""")
