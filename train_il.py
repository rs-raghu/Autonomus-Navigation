"""
train_il.py  —  Phase 3: Imitation Learning (MLP warm-start)

Loads all expert CSVs from reports/, trains a 4-input MLP on the Pure Pursuit
demonstration data, saves the model + normalisation stats, and writes plots.

Run:
    python train_il.py

Outputs
-------
  model_il.pt          PyTorch checkpoint  (model weights + norm stats)
  reports/il_loss.png  Training + validation loss curve
  reports/il_cte.png   CTE histogram (train vs val split)
  reports/il_pred.png  Predicted vs actual steer_input scatter (val set)

Feature vector (4 inputs)
--------------------------
  cte_px              signed cross-track error (px)
  heading_error_deg   heading vs path tangent (degrees)
  speed_px_s          tractor speed (px / s)
  lookahead_angle_deg Pure Pursuit α at the logged pose (degrees)

Label (1 output)
----------------
  steer_input         normalised steering [-1, 1]

Split strategy
--------------
Rows are split by lap number, NOT by random row shuffle.  Consecutive rows are
highly correlated (same lap, adjacent positions), so a random split leaks future
positions into training and inflates val accuracy.  The last VAL_LAP_FRAC of
unique lap numbers form the validation set.
"""

import os
import glob
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────

FEATURE_COLS  = ["cte_px", "heading_error_deg", "speed_px_s", "lookahead_angle_deg"]
LABEL_COL     = "steer_input"

VAL_LAP_FRAC  = 0.20     # last 20 % of unique laps → validation
BATCH_SIZE    = 512
EPOCHS        = 80
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 12        # early-stopping patience (epochs)

MODEL_OUT     = "model_il.pt"
REPORT_DIR    = "reports"

SEED          = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ──────────────────────────────────────────────────────────────────────

class SteeringMLP(nn.Module):
    """
    3 hidden layers, tanh activations throughout.
    Output tanh squashes prediction into [-1, 1] — the same range as steer_input.
    """

    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),     nn.Tanh(),
            nn.Linear(64, 32),     nn.Tanh(),
            nn.Linear(32, 1),      nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # shape (B,)


# ── Dataset ────────────────────────────────────────────────────────────────────

class SteeringDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csvs() -> pd.DataFrame:
    """
    Load every expert_*.csv (and optionally trial_*.csv) from reports/.
    Both share the same FEATURE_COLS + LABEL_COL schema.
    Drops rows with NaN in any required column.
    """
    patterns = [
        os.path.join(REPORT_DIR, "expert_*.csv"),
        os.path.join(REPORT_DIR, "trial_*.csv"),
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{REPORT_DIR}/'.  "
            f"Run phase2_expert.py first to generate expert_1.csv."
        )

    dfs: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp, usecols=FEATURE_COLS + [LABEL_COL, "lap"])
        before = len(df)
        df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
        dropped = before - len(df)
        tag = "expert" if "expert" in os.path.basename(fp) else "trial"
        print(f"  [{tag}] {os.path.basename(fp):25s}  "
              f"{len(df):>7,} rows  (dropped {dropped} NaN rows)")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total: {len(combined):,} rows from {len(files)} file(s)")
    return combined


def lap_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by lap number.  Last VAL_LAP_FRAC of unique laps → val.
    Returns (train_df, val_df).
    """
    laps     = sorted(df["lap"].unique())
    n_val    = max(1, int(len(laps) * VAL_LAP_FRAC))
    val_laps = set(laps[-n_val:])
    train_df = df[~df["lap"].isin(val_laps)].copy()
    val_df   = df[ df["lap"].isin(val_laps)].copy()
    print(f"\n  Lap split:  train laps 1–{laps[-n_val-1]}  "
          f"({len(train_df):,} rows)   "
          f"val laps {laps[-n_val]}–{laps[-1]}  ({len(val_df):,} rows)")
    return train_df, val_df


# ── Normalisation ──────────────────────────────────────────────────────────────

def compute_norm_stats(train_df: pd.DataFrame) -> dict[str, list[float]]:
    """Compute per-feature mean and std on the TRAINING set only."""
    mean = train_df[FEATURE_COLS].mean().tolist()
    std  = train_df[FEATURE_COLS].std().tolist()
    # Guard against zero-variance features (e.g. speed if tractor was always
    # at the same throttle)
    std  = [max(s, 1e-6) for s in std]
    return {"mean": mean, "std": std}


def normalise(df: pd.DataFrame, stats: dict) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    X    = df[FEATURE_COLS].values.astype(np.float32)
    return (X - mean) / std


# ── Training ───────────────────────────────────────────────────────────────────

def train(model: SteeringMLP,
          train_loader: DataLoader,
          val_loader:   DataLoader) -> tuple[list[float], list[float]]:

    optimiser  = torch.optim.Adam(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=5)
    criterion  = nn.MSELoss()

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss = float("inf")
    patience_ctr  = 0

    print(f"\n  Training on {device}  —  {EPOCHS} epochs max, "
          f"early stop patience {PATIENCE}\n")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimiser.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(y_batch)

        train_loss = running_loss / len(train_loader.dataset)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred     = model(X_batch)
                running_val += criterion(pred, y_batch).item() * len(y_batch)
        val_loss = running_val / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={optimiser.param_groups[0]['lr']:.2e}")

        # ── Early stopping ─────────────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), "_best_weights.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch}  "
                      f"(best val MSE = {best_val_loss:.5f})")
                break

    # Restore best weights
    model.load_state_dict(torch.load("_best_weights.pt", map_location=device, weights_only=True))
    os.remove("_best_weights.pt")
    return train_losses, val_losses


# ── Save checkpoint ────────────────────────────────────────────────────────────

def save_checkpoint(model: SteeringMLP, norm_stats: dict,
                    train_losses: list[float], val_losses: list[float]) -> None:
    checkpoint = {
        "model_state":  model.state_dict(),
        "norm_stats":   norm_stats,           # mean / std for inference
        "feature_cols": FEATURE_COLS,
        "label_col":    LABEL_COL,
        "train_loss":   train_losses[-1],
        "val_loss":     val_losses[-1],
        "best_val_mse": min(val_losses),
        "best_val_rmse": math.sqrt(min(val_losses)),
    }
    torch.save(checkpoint, MODEL_OUT)
    print(f"\n  Checkpoint saved → {MODEL_OUT}")
    print(f"  Best val MSE  = {min(val_losses):.5f}")
    print(f"  Best val RMSE = {math.sqrt(min(val_losses)):.5f}  "
          f"(steer units, range ±1)")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_loss_curve(train_losses: list[float],
                    val_losses:   list[float]) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train MSE", linewidth=1.5)
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Imitation learning — training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(REPORT_DIR, "il_loss.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Loss curve     → {out}")


def plot_cte_histogram(train_df: pd.DataFrame,
                       val_df:   pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-80, 80, 60)
    ax.hist(train_df["cte_px"], bins=bins, alpha=0.6, label="Train", density=True)
    ax.hist(val_df["cte_px"],   bins=bins, alpha=0.6, label="Val",   density=True)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("CTE (px)  [+ = right of path]")
    ax.set_ylabel("Density")
    ax.set_title("Cross-track error distribution — train vs val")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(REPORT_DIR, "il_cte.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  CTE histogram  → {out}")


def plot_pred_vs_actual(model:     SteeringMLP,
                        val_loader: DataLoader,
                        val_df:    pd.DataFrame) -> None:
    model.eval()
    preds: list[float] = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            preds.extend(model(X_batch.to(device)).cpu().numpy().tolist())

    actual = val_df[LABEL_COL].values[: len(preds)]
    preds  = np.array(preds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter
    axes[0].scatter(actual, preds, s=1, alpha=0.3, rasterized=True)
    axes[0].plot([-1, 1], [-1, 1], "r--", linewidth=1, label="Perfect")
    axes[0].set_xlabel("Actual steer_input")
    axes[0].set_ylabel("Predicted steer_input")
    axes[0].set_title("Predicted vs actual (val set)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)

    # Residuals histogram
    residuals = preds - actual
    axes[1].hist(residuals, bins=80, density=True)
    axes[1].set_xlabel("Residual  (pred − actual)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Residuals  (RMSE = {math.sqrt(np.mean(residuals**2)):.4f})")
    axes[1].grid(True, alpha=0.3)

    out = os.path.join(REPORT_DIR, "il_pred.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Pred vs actual → {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("  Phase 3 — Imitation Learning")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading CSVs …")
    df          = load_csvs()
    train_df, val_df = lap_split(df)

    # Normalise
    print("\n[2/5] Computing normalisation stats …")
    norm_stats  = compute_norm_stats(train_df)
    for name, mean, std in zip(FEATURE_COLS, norm_stats["mean"], norm_stats["std"]):
        print(f"  {name:25s}  mean={mean:+9.3f}  std={std:.3f}")

    X_train = normalise(train_df, norm_stats)
    y_train = train_df[LABEL_COL].values.astype(np.float32)
    X_val   = normalise(val_df,   norm_stats)
    y_val   = val_df[LABEL_COL].values.astype(np.float32)

    train_dataset = SteeringDataset(X_train, y_train)
    val_dataset   = SteeringDataset(X_val,   y_val)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  drop_last=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    # Build model
    print("\n[3/5] Building model …")
    model = SteeringMLP(in_dim=len(FEATURE_COLS)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Train
    print("\n[4/5] Training …")
    train_losses, val_losses = train(model, train_loader, val_loader)

    # Save
    print("\n[5/5] Saving checkpoint and plots …")
    save_checkpoint(model, norm_stats, train_losses, val_losses)
    plot_loss_curve(train_losses, val_losses)
    plot_cte_histogram(train_df, val_df)
    plot_pred_vs_actual(model, val_loader, val_df)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s.  "
          f"Load the policy with phase3_il.py.\n")


if __name__ == "__main__":
    main()