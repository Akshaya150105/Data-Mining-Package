"""
stgcn_train.py
==============
Trains a Spatio-Temporal Graph Convolutional Network (STGCN)
on your Aadhaar enrolment data.

Reads:
    adjacency_output/feature_tensor_X.npy    [T=70, N=945, C=7]
    adjacency_output/L_normalised_laplacian.csv  [945 x 945]
    adjacency_output/district_order.csv      district names

Outputs (in stgcn_output/):
    best_model.pt          best checkpoint (lowest val loss)
    loss_curve.png         train + val loss over epochs
    predictions.npy        model predictions on test set [T_test, N, C]
    actuals.npy            ground truth on test set
    metrics.txt            MAE, RMSE, MAPE per feature
    per_district_error.csv MAE per district (for choropleth)
    pred_vs_actual.png     sample districts predicted vs actual

Usage:
    python stgcn_train.py
    python stgcn_train.py --epochs 200 --lr 0.0005 --t_in 12

Dependencies:
    pip install torch numpy pandas matplotlib scikit-learn
    (no torch_geometric needed — pure PyTorch implementation)
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path("stgcn_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BG = "#f9f8f6"


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data(tensor_path, laplacian_path, district_path):
    X = np.load(tensor_path).astype(np.float32)          # [T, N, C]
    L_df = pd.read_csv(laplacian_path, index_col=0)
    L = L_df.values.astype(np.float32)                   # [N, N]
    districts_df = pd.read_csv(district_path)
    # district_order.csv has index col + district col
    districts = (districts_df["district"].tolist()
                 if "district" in districts_df.columns
                 else districts_df.iloc[:, 1].tolist())

    T, N, C = X.shape
    print(f"  Tensor X: T={T}, N={N}, C={C}")
    print(f"  Laplacian: {L.shape}")
    print(f"  Districts: {len(districts)}")

    # Scaled Laplacian for Chebyshev: L_tilde = 2L/lambda_max - I
    lambda_max = np.linalg.eigvalsh(L).max()
    L_tilde = (2.0 / (lambda_max + 1e-8)) * L - np.eye(N, dtype=np.float32)

    return X, L_tilde, districts, T, N, C


def make_sequences(X, t_in, t_out=1):
    """
    Sliding window: X[t:t+t_in] -> X[t+t_in:t+t_in+t_out]
    Returns (inputs [S, t_in, N, C], targets [S, t_out, N, C])
    Larger t_out gives more supervision signal per sequence.
    """
    T = X.shape[0]
    inputs, targets = [], []
    for t in range(T - t_in - t_out + 1):
        inputs.append(X[t : t + t_in])
        targets.append(X[t + t_in : t + t_in + t_out])
    return np.array(inputs), np.array(targets)


def z_score_normalise(X):
    """
    Per-feature z-score normalisation across (T, N) dimensions.
    Prevents large-volume districts from dominating the loss.
    Returns (X_norm, mean, std) for denormalisation later.
    """
    # X shape: [T, N, C]
    mean = X.mean(axis=(0, 1), keepdims=True)   # [1, 1, C]
    std  = X.std(axis=(0, 1), keepdims=True)     # [1, 1, C]
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean.squeeze(), std.squeeze()



def train_val_test_split(inputs, targets, train_r=0.6, val_r=0.2):
    S = len(inputs)
    t1 = int(S * train_r)
    t2 = int(S * (train_r + val_r))
    return (inputs[:t1],  targets[:t1],
            inputs[t1:t2], targets[t1:t2],
            inputs[t2:],   targets[t2:])


# ══════════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════

class ChebConv(nn.Module):
    """
    Chebyshev spectral graph convolution.
    Approximates the graph convolution with K-hop polynomial filters.
    Input:  x  [B, T, N, C_in]
    Output:    [B, T, N, C_out]
    """
    def __init__(self, c_in, c_out, K, L_tilde):
        super().__init__()
        self.K       = K
        self.c_in    = c_in
        self.c_out   = c_out
        # Register Laplacian as buffer (moved to device automatically)
        self.register_buffer("L_tilde",
                             torch.tensor(L_tilde, dtype=torch.float32))
        # Learnable weights: one matrix per Chebyshev order
        self.weight = nn.Parameter(
            torch.FloatTensor(K, c_in, c_out))
        self.bias   = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [B, T, N, C_in]
        B, T, N, C = x.shape
        out = torch.zeros(B, T, N, self.c_out, device=x.device)

        x_flat = x.reshape(B * T, N, C)            # [BT, N, C_in]

        Tx_0 = x_flat                               # T_0(L)x = x
        Tx_1 = torch.bmm(
            self.L_tilde.unsqueeze(0).expand(B*T,-1,-1),
            x_flat)                                 # T_1(L)x = L_tilde @ x

        out_flat = torch.einsum('bnc,co->bno', Tx_0, self.weight[0])
        if self.K > 1:
            out_flat = out_flat + torch.einsum('bnc,co->bno', Tx_1, self.weight[1])

        for k in range(2, self.K):
            Tx_2 = 2 * torch.bmm(
                self.L_tilde.unsqueeze(0).expand(B*T,-1,-1),
                Tx_1) - Tx_0
            out_flat = out_flat + torch.einsum('bnc,co->bno', Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        out_flat = out_flat + self.bias
        return out_flat.reshape(B, T, N, self.c_out)


class TemporalConv(nn.Module):
    """
    Gated 1-D temporal convolution (GLU activation).
    Input:  [B, T, N, C_in]
    Output: [B, T - (Kt-1), N, C_out]
    """
    def __init__(self, c_in, c_out, Kt):
        super().__init__()
        self.Kt   = Kt
        self.conv = nn.Conv2d(c_in, 2 * c_out,
                              kernel_size=(Kt, 1))
        self.bn   = nn.BatchNorm2d(2 * c_out)

    def forward(self, x):
        x   = x.permute(0, 3, 1, 2)        # [B, C, T, N]
        out = self.bn(self.conv(x))         # [B, 2*C_out, T', N]
        P, Q = out.chunk(2, dim=1)
        out  = P * torch.sigmoid(Q)         # GLU gate
        return out.permute(0, 2, 3, 1)     # [B, T', N, C_out]


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Conv Block: Temporal -> Graph -> BN -> Temporal
    with residual connection and layer normalisation.
    """
    def __init__(self, c_in, c_spatial, c_out, Kt, K, L_tilde):
        super().__init__()
        self.t1    = TemporalConv(c_in,      c_out,     Kt)
        self.gc    = ChebConv(c_out,         c_spatial, K, L_tilde)
        self.gc_bn = nn.BatchNorm1d(c_spatial)
        self.t2    = TemporalConv(c_spatial, c_out,     Kt)
        self.norm  = nn.LayerNorm(c_out)
        self.res   = nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.1)

    def forward(self, x):
        T_in = x.shape[1]
        B, _, N, _ = x.shape
        res  = x

        out = self.t1(x)                        # [B, T-Kt+1, N, C_out]
        gc  = self.gc(out)                      # [B, T-Kt+1, N, c_spatial]

        # BatchNorm1d expects [B*T, C, N] or [B*T*N, C]
        BT, T2, Ng, Cs = gc.shape
        gc_flat = gc.reshape(BT * T2, Ng, Cs).reshape(BT * T2 * Ng, Cs)
        # Use LayerNorm instead of BN for variable-length sequences
        gc = self.relu(gc)
        gc = self.drop(gc)

        out = self.t2(gc)                       # [B, T-2*(Kt-1), N, C_out]

        T_out = out.shape[1]
        trim  = T_in - T_out
        res   = res[:, trim//2 : T_in - (trim - trim//2), :, :]
        res   = self.res(res)

        return self.norm(out + res)


class STGCN(nn.Module):
    """
    Full STGCN:
      - 2 ST-Conv blocks
      - Output FC layer predicting next step
    """
    def __init__(self, C, N, T_in, T_out, K, Kt, L_tilde,
                 c_s=16, c_out=64):
        super().__init__()
        self.t_out_steps = T_out          # store for slicing in forward()
        self.block1 = STConvBlock(C,     c_s, c_out, Kt, K, L_tilde)
        self.block2 = STConvBlock(c_out, c_s, c_out, Kt, K, L_tilde)
        self.t_out  = TemporalConv(c_out, c_out, Kt)
        self.fc     = nn.Linear(c_out, C)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, T_in, N, C]
        out = self.block1(x)
        out = self.dropout(out)
        out = self.block2(out)
        out = self.t_out(out)            # [B, T_remaining, N, c_out]
        out = self.fc(out)               # [B, T_remaining, N, C]
        # Slice last t_out steps — model produces T_remaining >= t_out
        out = out[:, -self.t_out_steps:, :, :]
        return out


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train(model, optimizer, scheduler, inputs, targets,
          val_in, val_tgt, epochs, patience):

    L_train, L_val = [], []
    best_val   = float("inf")
    no_improve = 0
    best_state = None

    inp_t  = torch.tensor(inputs,  device=DEVICE)   # [S, T_in, N, C]
    tgt_t  = torch.tensor(targets, device=DEVICE)   # [S, 1,    N, C]
    vinp_t = torch.tensor(val_in,  device=DEVICE)
    vtgt_t = torch.tensor(val_tgt, device=DEVICE)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred  = model(inp_t)
        loss  = F.mse_loss(pred, tgt_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vpred = model(vinp_t)
            vloss = F.mse_loss(vpred, vtgt_t).item()

        L_train.append(loss.item())
        L_val.append(vloss)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train={loss.item():.6f}  val={vloss:.6f}  "
                  f"lr={lr_now:.6f}")

        if vloss < best_val - 1e-6:
            best_val   = vloss
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    return L_train, L_val, best_state


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def smape(actual, predicted):
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    mask  = denom > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)

def masked_mape(actual, predicted, threshold=0.05):
    mask = np.abs(actual) > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def r2_score(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)

def evaluate(model, test_in, test_tgt, districts, C):
    model.eval()
    inp_t = torch.tensor(test_in, device=DEVICE)
    with torch.no_grad():
        preds = model(inp_t).cpu().numpy()
    actuals = test_tgt

    P = preds[:, 0, :, :].reshape(-1, C)
    A = actuals[:, 0, :, :].reshape(-1, C)

    feature_names = [
        "enrol_total", "bio_total", "bio_age5_ratio",
        "enrol_minor_ratio", "enrol_adult_ratio",
        "bio_dependency", "enrol_growth_pct (noisy)",
    ][:C]

    header = "  {:<28} {:>8} {:>8} {:>9} {:>7} {:>9}".format(
        "Feature", "MAE", "RMSE", "sMAPE", "R2", "mMAPE")
    sep = "  " + "-" * 72

    lines_out = [
        "=" * 60,
        "STGCN EVALUATION RESULTS",
        "=" * 60,
        "",
        "All metrics on MinMax-normalised [0,1] scale.",
        "  MAE / RMSE : absolute error — lower is better",
        "  sMAPE      : symmetric MAPE, robust to near-zero actuals [0..200%]",
        "  R2         : explained variance (1.0 = perfect, >0.7 = good)",
        "  mMAPE      : MAPE only where actual > 0.05 (avoids blow-up)",
        "",
        header, sep,
    ]

    maes, rmses, smapes, r2s, mmapes = [], [], [], [], []
    for i, fname in enumerate(feature_names):
        p_i  = P[:, i]
        a_i  = A[:, i]
        mae  = float(mean_absolute_error(a_i, p_i))
        rmse = float(np.sqrt(mean_squared_error(a_i, p_i)))
        sm   = smape(a_i, p_i)
        r2   = r2_score(a_i, p_i)
        mm   = masked_mape(a_i, p_i, threshold=0.05)
        maes.append(mae)
        rmses.append(rmse)
        smapes.append(sm)
        r2s.append(r2)
        if not np.isnan(mm):
            mmapes.append(mm)
        mm_s = "{:7.1f}%".format(mm) if not np.isnan(mm) else "    n/a"
        lines_out.append(
            "  {:<28} {:>8.4f} {:>8.4f} {:>8.1f}% {:>7.4f} {}".format(
                fname, mae, rmse, sm, r2, mm_s))

    mean_mm = float(np.mean(mmapes)) if mmapes else 0.0
    lines_out.append(sep)
    lines_out.append(
        "  {:<28} {:>8.4f} {:>8.4f} {:>8.1f}% {:>7.4f} {:>8.1f}%".format(
            "MEAN",
            float(np.mean(maes)), float(np.mean(rmses)),
            float(np.mean(smapes)), float(np.mean(r2s)), mean_mm))
    r2_mean = float(np.mean(r2s))
    if r2_mean > 0.6:
        r2_grade = "GOOD — model captures district variation well"
    elif r2_mean > 0.3:
        r2_grade = "MODERATE — model partially captures variation"
    else:
        r2_grade = "WEAK — model close to predicting mean"

    lines_out += [
        "",
        "INTERPRETATION:",
        "  MAE {:.4f} = predictions within {:.2f}% of the z-scored feature range.".format(
            float(np.mean(maes)), float(np.mean(maes)) * 100),
        "  sMAPE {:.1f}% — target <40% for strong performance.".format(float(np.mean(smapes))),
        "  R2 {:.4f} — {}.".format(r2_mean, r2_grade),
        "  Note: enrol_growth_pct is inherently noisy; exclude from R2 average for cleaner signal.",
        "  Substantive R2 (ex growth): {:.4f}".format(float(np.mean(r2s[:-1]))),
    ]

    report = "\n".join(lines_out)
    print(report)
    (OUTPUT_DIR / "metrics.txt").write_text(report)

    N      = len(districts)
    p_dist = preds[:, 0, :, 0]
    a_dist = actuals[:, 0, :, 0]

    dist_mae   = np.mean(np.abs(p_dist - a_dist), axis=0)
    dist_smape = np.array([smape(a_dist[:, di], p_dist[:, di])
                           for di in range(N)])
    dist_r2    = np.array([r2_score(a_dist[:, di], p_dist[:, di])
                           for di in range(N)])

    dist_df = pd.DataFrame({
        "district": districts[:N],
        "mae":      dist_mae,
        "smape":    dist_smape,
        "r2":       dist_r2,
    }).sort_values("mae", ascending=False)

    dist_df.to_csv(OUTPUT_DIR / "per_district_error.csv", index=False)
    print("\nTop 10 districts by MAE (enrol_total):")
    print(dist_df.head(10)[["district", "mae", "smape", "r2"]].to_string(index=False))

    return preds, actuals, dist_df


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_loss(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor("#ffffff")
    ax.plot(train_losses, color="#534AB7", linewidth=2,   label="Train loss")
    ax.plot(val_losses,   color="#1D9E75", linewidth=2,   label="Val loss")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("MSE loss", fontsize=10)
    ax.set_title("STGCN training — loss curve", fontsize=13,
                 fontweight="bold", loc="left", pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, color="#f0f0f0")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#e5e7eb")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → loss_curve.png")


def plot_predictions(preds, actuals, districts, n_districts=6):
    """Plot predicted vs actual enrol_total for n sample districts."""
    N   = preds.shape[2]
    idx = np.random.choice(N, n_districts, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), facecolor=BG)
    axes = axes.flatten()

    for i, di in enumerate(idx):
        ax = axes[i]
        ax.set_facecolor("#ffffff")
        p = preds[:, 0, di, 0]    # predicted enrol_total
        a = actuals[:, 0, di, 0]  # actual
        ax.plot(a, color="#534AB7", linewidth=2,   label="Actual")
        ax.plot(p, color="#D85A30", linewidth=2,
                linestyle="--", label="Predicted")
        mae_d = np.mean(np.abs(p - a))
        name  = districts[di][:18] if di < len(districts) else f"District {di}"
        ax.set_title(f"{name}\nMAE={mae_d:.4f}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax.spines[sp].set_color("#e5e7eb")
        ax.grid(True, color="#f0f0f0", linewidth=0.6)

    plt.suptitle("STGCN — predicted vs actual enrolment (test set, 6 sample districts)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pred_vs_actual.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → pred_vs_actual.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 60)
    print("STGCN TRAINING — AADHAAR ENROLMENT")
    print("=" * 60)
    print(f"  T_in={args.t_in}  T_out={args.t_out}  K={args.k}  Kt={args.kt}")
    print(f"  c_spatial={args.c_spatial}  c_out={args.c_out}")
    print(f"  Loss: MSE")
    print(f"  epochs={args.epochs}  lr={args.lr}  patience={args.patience}")

    # ── Load data ──────────────────────────────────────────────────────
    print("\n── Loading data ──")
    X, L_tilde, districts, T, N, C = load_data(
        args.tensor, args.laplacian, args.districts
    )

    # ── Sliding window sequences ───────────────────────────────────────
    print("\n── Normalising data (z-score per feature) ──")
    X_norm, feat_mean, feat_std = z_score_normalise(X)
    np.save(OUTPUT_DIR / "feat_mean.npy", feat_mean)
    np.save(OUTPUT_DIR / "feat_std.npy",  feat_std)
    print(f"  Feature means: {feat_mean.round(4)}")
    print(f"  Feature stds:  {feat_std.round(4)}")

    print("\n── Creating sequences ──")
    inputs, targets = make_sequences(X_norm, args.t_in, t_out=args.t_out)
    print(f"  Sequences: inputs={inputs.shape}  targets={targets.shape}")

    tr_in, tr_tgt, va_in, va_tgt, te_in, te_tgt = train_val_test_split(
        inputs, targets, train_r=0.6, val_r=0.2
    )
    print(f"  Train={len(tr_in)}  Val={len(va_in)}  Test={len(te_in)}")

    # ── Build model ────────────────────────────────────────────────────
    print("\n── Building STGCN ──")
    # Validate: with 2 ST blocks + 1 output conv, each with Kt, total
    # temporal reduction = 5*(Kt-1). T_in must be > 5*(Kt-1) + t_out
    min_T = 5 * (args.kt - 1) + args.t_out
    if args.t_in <= min_T:
        print(f"  WARNING: T_in={args.t_in} too small for Kt={args.kt}, "
              f"t_out={args.t_out} (need T_in > {min_T}). "
              f"Auto-adjusting T_in to {min_T + 2}")
        args.t_in = min_T + 2
        inputs, targets = make_sequences(X_norm, args.t_in, t_out=args.t_out)
        tr_in, tr_tgt, va_in, va_tgt, te_in, te_tgt = train_val_test_split(
            inputs, targets, train_r=0.6, val_r=0.2)
        print(f"  Recomputed: Train={len(tr_in)} Val={len(va_in)} Test={len(te_in)}")
    model = STGCN(
        C=C, N=N, T_in=args.t_in, T_out=args.t_out,
        K=args.k, Kt=args.kt,
        L_tilde=L_tilde,
        c_s=args.c_spatial, c_out=args.c_out,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    # Cosine annealing with warm restarts — better for small datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(10, args.epochs // 4), T_mult=1, eta_min=1e-5)

    # ── Train ──────────────────────────────────────────────────────────
    print(f"\n── Training ({args.epochs} epochs max) ──")
    train_losses, val_losses, best_state = train(
        model, optimizer, scheduler,
        tr_in, tr_tgt, va_in, va_tgt,
        epochs=args.epochs, patience=args.patience,
    )

    # Save best checkpoint
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": best_state,
        "config": vars(args),
        "T": T, "N": N, "C": C,
    }, OUTPUT_DIR / "best_model.pt")
    print(f"  Best model saved → best_model.pt")

    # ── Evaluate ───────────────────────────────────────────────────────
    print("\n── Evaluating on test set ──")
    preds, actuals, dist_errors = evaluate(
        model, te_in, te_tgt, districts, C)

    # Save raw arrays
    np.save(OUTPUT_DIR / "predictions.npy", preds)
    np.save(OUTPUT_DIR / "actuals.npy",     actuals)

    # ── Plots ──────────────────────────────────────────────────────────
    print("\n── Saving plots ──")
    plot_loss(train_losses, val_losses)
    plot_predictions(preds, actuals, districts)

    print("\n" + "=" * 60)
    print("DONE — stgcn_output/")
    print("  best_model.pt         best checkpoint")
    print("  loss_curve.png        train + val loss")
    print("  pred_vs_actual.png    sample district predictions")
    print("  per_district_error.csv  MAE per district")
    print("  predictions.npy / actuals.npy  raw arrays")
    print("  metrics.txt           MAE, RMSE, MAPE per feature")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train STGCN on Aadhaar data")
    p.add_argument("--tensor",     default="adjacency_output/feature_tensor_X.npy")
    p.add_argument("--laplacian",  default="adjacency_output/L_normalised_laplacian.csv")
    p.add_argument("--districts",  default="adjacency_output/district_order.csv")
    p.add_argument("--t_in",       type=int,   default=6,     help="Input window size")
    p.add_argument("--t_out",      type=int,   default=1,     help="Prediction steps ahead")
    p.add_argument("--k",          type=int,   default=4,     help="Chebyshev order (4 = 4-hop neighbourhood)")
    p.add_argument("--kt",         type=int,   default=2,     help="Temporal kernel size (2 = less shrinkage)")
    p.add_argument("--c_spatial",  type=int,   default=16,    help="Spatial channels")
    p.add_argument("--c_out",      type=int,   default=64,    help="Output channels")
    p.add_argument("--epochs",     type=int,   default=200,  help="Max epochs")
    p.add_argument("--lr",         type=float, default=0.001, help="Initial learning rate")
    p.add_argument("--patience",   type=int,   default=25,    help="Early stopping patience")
    args = p.parse_args()
    main(args)