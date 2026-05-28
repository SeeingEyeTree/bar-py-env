#!/usr/bin/env python3
"""Train a behavioral cloning model on the BC dataset built by build_bc_dataset.py.

Architecture: scalar MLP branch + spatial CNN branch → 3 output heads:
  1. action_type  (9-class CrossEntropy)
  2. build_def    (41-class CrossEntropy, masked to BUILD_* steps)
  3. target_xz    (2-float MSE + Sigmoid, masked to non-NO_OP steps with positions)

Saves bc_model.pt (PyTorch weights) and bc_train_stats.json.

Usage:
    python scripts/train_bc.py
    python scripts/train_bc.py --dataset D:\\BAR_Replays\\training_data\\bc_dataset.npz
    python scripts/train_bc.py --epochs 150 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env.features import ACTION_NAMES, N_ACTIONS, GRID_H, GRID_W, N_SCALARS

DEFAULT_DATASET = Path(r"D:\BAR_Replays\training_data\bc_dataset.npz")
DEFAULT_MODEL_OUT = PACKAGE_ROOT / "bc_model.pt"
DEFAULT_STATS_OUT = PACKAGE_ROOT / "bc_train_stats.json"

N_BUILD_DEF = 41   # top-40 defs + "other"
BUILD_TYPES  = {1, 2, 3, 4, 5}  # action_types that use build_def head


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    def __init__(
        self,
        scalars:       np.ndarray,  # (N, N_SCALARS)
        spatial:       np.ndarray,  # (N, 7, GRID_H, GRID_W)
        action_type:   np.ndarray,  # (N,) int32
        build_def_idx: np.ndarray,  # (N,) int32, -1 = N/A
        target_xz:     np.ndarray,  # (N, 2) float32, (-1,-1) = N/A
        sc_mean:       np.ndarray,
        sc_std:        np.ndarray,
    ) -> None:
        self.scalars       = torch.from_numpy((scalars - sc_mean) / sc_std).float()
        self.spatial       = torch.from_numpy(spatial).float()
        self.action_type   = torch.from_numpy(action_type).long()
        self.build_def_idx = torch.from_numpy(build_def_idx).long()
        self.target_xz     = torch.from_numpy(target_xz).float()

    def __len__(self) -> int:
        return len(self.action_type)

    def __getitem__(self, idx: int):
        return (
            self.scalars[idx],
            self.spatial[idx],
            self.action_type[idx],
            self.build_def_idx[idx],
            self.target_xz[idx],
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleBCModel(nn.Module):
    """Scalar MLP + spatial CNN → 3 output heads for behavioral cloning."""

    def __init__(
        self,
        n_scalars: int = N_SCALARS,
        grid_h: int = GRID_H,
        grid_w: int = GRID_W,
        n_actions: int = N_ACTIONS,
        n_build_def: int = N_BUILD_DEF,
    ) -> None:
        super().__init__()

        # Scalar branch
        self.scalar_net = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )  # → (64,)

        # Spatial branch
        self.spatial_net = nn.Sequential(
            nn.Conv2d(7, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # → (32, 24, 32)
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # → (32, 12, 16)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (grid_h // 4) * (grid_w // 4), 128),
            nn.ReLU(),
        )  # → (128,)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )  # → (128,)

        # Heads
        self.action_head    = nn.Linear(128, n_actions)
        self.build_def_head = nn.Linear(128, n_build_def)
        self.target_pos_head = nn.Sequential(
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        scalars: torch.Tensor,   # (B, N_SCALARS)
        spatial: torch.Tensor,   # (B, 7, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_s  = self.scalar_net(scalars)
        z_sp = self.spatial_net(spatial)
        z    = self.fusion(torch.cat([z_s, z_sp], dim=1))
        return (
            self.action_head(z),      # (B, N_ACTIONS)
            self.build_def_head(z),   # (B, N_BUILD_DEF)
            self.target_pos_head(z),  # (B, 2)
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _compute_stats(scalars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = scalars.mean(axis=0)
    std  = scalars.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def _run_epoch(
    model: SimpleBCModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = total_l_action = total_l_build = total_l_pos = 0.0
    correct_action = total_action = 0
    n_build = n_build_correct = 0
    n_steps = 0

    for sc, sp, at, bd, txz in loader:
        sc  = sc.to(device)
        sp  = sp.to(device)
        at  = at.to(device)
        bd  = bd.to(device)
        txz = txz.to(device)

        if training:
            optimizer.zero_grad()

        logits_at, logits_bd, pred_pos = model(sc, sp)

        # Head 1: action_type (with optional class weights to handle imbalance)
        l_action = F.cross_entropy(logits_at, at, weight=class_weights)

        # Head 2: build_def (masked to BUILD_* steps with valid vocab index)
        build_mask = torch.zeros(len(at), dtype=torch.bool, device=device)
        for bt in BUILD_TYPES:
            build_mask |= (at == bt)
        build_mask &= (bd >= 0)
        if build_mask.sum() > 0:
            l_build = F.cross_entropy(logits_bd[build_mask], bd[build_mask])
        else:
            l_build = torch.tensor(0.0, device=device)

        # Head 3: target_xz (masked to steps with valid positions)
        pos_mask = (txz[:, 0] >= 0)
        if pos_mask.sum() > 0:
            l_pos = F.mse_loss(pred_pos[pos_mask], txz[pos_mask])
        else:
            l_pos = torch.tensor(0.0, device=device)

        loss = l_action + 0.5 * l_build + 0.1 * l_pos

        if training:
            loss.backward()
            optimizer.step()

        # Metrics
        total_loss     += loss.item()
        total_l_action += l_action.item()
        total_l_build  += l_build.item()
        total_l_pos    += l_pos.item()
        n_steps        += 1

        preds = logits_at.argmax(dim=1)
        correct_action += (preds == at).sum().item()
        total_action   += len(at)

        if build_mask.sum() > 0:
            bd_preds = logits_bd[build_mask].argmax(dim=1)
            n_build_correct += (bd_preds == bd[build_mask]).sum().item()
            n_build += int(build_mask.sum())

    return {
        "loss":           total_loss / max(n_steps, 1),
        "loss_action":    total_l_action / max(n_steps, 1),
        "loss_build":     total_l_build / max(n_steps, 1),
        "loss_pos":       total_l_pos / max(n_steps, 1),
        "acc_action":     correct_action / max(total_action, 1),
        "acc_build_def":  n_build_correct / max(n_build, 1),
    }


def _confusion_matrix(
    model: SimpleBCModel,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> np.ndarray:
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for sc, sp, at, *_ in loader:
            preds = model(sc.to(device), sp.to(device))[0].argmax(dim=1).cpu().numpy()
            for p, t in zip(preds, at.numpy()):
                cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",   type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out",       type=Path, default=DEFAULT_MODEL_OUT)
    ap.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_OUT)
    ap.add_argument("--epochs",    type=int,  default=100)
    ap.add_argument("--batch",     type=int,  default=128)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--val-games", type=int,  default=3,
                    help="Number of replay files to reserve for validation (by game_id).")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}\nRun build_bc_dataset.py first.", file=sys.stderr)
        return 1

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print(f"Loading dataset: {args.dataset}")
    data = np.load(args.dataset)
    scalars       = data["scalars"]        # (N, N_SCALARS)
    spatial       = data["spatial"]        # (N, 7, H, W)
    action_type   = data["action_type"]    # (N,)
    build_def_idx = data["build_def_idx"]  # (N,)
    target_xz     = data["target_xz"]     # (N, 2)
    game_id       = data["game_id"]        # (N,)
    build_def_vocab = data["build_def_vocab"]  # (41,)
    N = len(scalars)
    print(f"  {N} samples  spatial={spatial.shape[1:]}")

    # Train / val split by game_id
    max_game = int(game_id.max())
    val_game_ids = set(range(max(0, max_game - args.val_games + 1), max_game + 1))
    train_mask = np.array([g not in val_game_ids for g in game_id])
    val_mask   = ~train_mask
    print(f"  Train: {train_mask.sum()} samples  Val: {val_mask.sum()} samples  "
          f"(val game_ids: {sorted(val_game_ids)})")

    # Normalise scalars using train statistics
    sc_mean, sc_std = _compute_stats(scalars[train_mask])

    def _make_ds(mask: np.ndarray) -> BCDataset:
        return BCDataset(
            scalars[mask], spatial[mask], action_type[mask],
            build_def_idx[mask], target_xz[mask],
            sc_mean, sc_std,
        )

    train_ds = _make_ds(train_mask)
    val_ds   = _make_ds(val_mask)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # Majority-class baseline for action_type
    majority_class = int(np.bincount(action_type[train_mask]).argmax())
    majority_acc   = (action_type[val_mask] == majority_class).mean()
    print(f"\nMajority-class baseline on val: {ACTION_NAMES[majority_class]} -> "
          f"acc={majority_acc:.3f}")

    # Class weights: pure inverse-frequency so every class contributes equally
    # to the gradient. Small upper clip only (prevents extreme BUILD_DEFENSE weight).
    # NO_OP is hard-set to 0.02 — it represents human idle/reaction lag, not
    # a strategic behaviour worth imitating.
    counts = np.bincount(action_type[train_mask], minlength=N_ACTIONS).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    raw_w  = 1.0 / counts
    raw_w  = np.clip(raw_w / raw_w.mean(), 0.001, 8.0)
    raw_w[0] = 0.02   # NO_OP hard override: suppress human idle frames
    print("Class weights:")
    for i, (name, w) in enumerate(zip(ACTION_NAMES, raw_w)):
        print(f"  {i}  {name:<16}  {w:.3f}")
    print()

    # -----------------------------------------------------------------------
    # Model + optimiser
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = SimpleBCModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")
    class_weights = torch.from_numpy(raw_w).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_acc = 0.0
    best_state   = None

    print(f"{'Ep':>4}  {'TrainLoss':>9}  {'ValLoss':>8}  {'TrainAcc':>8}  "
          f"{'ValAcc':>7}  {'BuildAcc':>8}")
    print("-" * 58)

    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, optimizer, device, class_weights)
        vl = _run_epoch(model, val_loader,   None,      device, class_weights)
        scheduler.step()

        if vl["acc_action"] > best_val_acc:
            best_val_acc = vl["acc_action"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch <= 5:
            print(f"{epoch:4d}  {tr['loss']:9.4f}  {vl['loss']:8.4f}  "
                  f"{tr['acc_action']:8.3f}  {vl['acc_action']:7.3f}  "
                  f"{vl['acc_build_def']:8.3f}")

    # -----------------------------------------------------------------------
    # Final eval with best weights
    # -----------------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    cm = _confusion_matrix(model, val_loader, device, N_ACTIONS)
    final = _run_epoch(model, val_loader, None, device)

    print(f"\n{'-'*58}")
    print(f"Best val action_type acc: {best_val_acc:.3f}  "
          f"(majority baseline: {majority_acc:.3f})")
    print(f"Val build_def acc:        {final['acc_build_def']:.3f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    header = "         " + " ".join(f"{n[:6]:>6}" for n in ACTION_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {ACTION_NAMES[i][:8]:<8} " + " ".join(f"{v:6d}" for v in row))

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    torch.save(
        {
            "model_state": model.state_dict(),
            "sc_mean": sc_mean,
            "sc_std":  sc_std,
            "build_def_vocab": build_def_vocab.tolist(),
            "action_names": ACTION_NAMES,
        },
        args.out,
    )
    print(f"\nModel saved: {args.out}")

    stats = {
        "val_acc_action":   float(best_val_acc),
        "val_acc_build_def": float(final["acc_build_def"]),
        "majority_baseline": float(majority_acc),
        "majority_class":    ACTION_NAMES[majority_class],
        "sc_mean": sc_mean.tolist(),
        "sc_std":  sc_std.tolist(),
        "build_def_vocab": build_def_vocab.tolist(),
        "action_names": ACTION_NAMES,
        "confusion_matrix": cm.tolist(),
        "n_train": int(train_mask.sum()),
        "n_val":   int(val_mask.sum()),
    }
    args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Stats saved: {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
