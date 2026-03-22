"""
stgcn_forecast.py
==================
Uses the trained STGCN checkpoint to autoregressively forecast
enrolment for the next N time steps beyond the dataset.

How it works:
  1. Loads best_model.pt + feature_tensor_X.npy
  2. Seeds the model with the LAST t_in real dates
  3. Autoregressively rolls forward: each prediction becomes
     part of the next input window
  4. Denormalises back to original scale using saved feat_mean/feat_std
  5. Saves per-district forecast + national/state aggregates

Outputs (in forecast_output/):
    forecast_raw.npy              raw predictions [n_steps, N, C]
    forecast.csv                  per-district forecast, all steps
    forecast_national.png         national total enrolment forecast
    forecast_top_states.png       top 10 states forecast trend
    forecast_map_step1.png        choropleth of step+1 predicted enrolment
    forecast_summary.csv          step-level national aggregates

Usage:
    python stgcn_forecast.py
    python stgcn_forecast.py --steps 12 --model stgcn_output/best_model.pt
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

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("forecast_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BG     = "#f9f8f6"

FEATURE_NAMES = [
    "enrol_total", "bio_total", "bio_age5_ratio",
    "enrol_minor_ratio", "enrol_adult_ratio",
    "bio_dependency", "enrol_growth_pct",
]


# ══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  (must match stgcn_train.py exactly)
# ══════════════════════════════════════════════════════════════════════════

class ChebConv(nn.Module):
    def __init__(self, c_in, c_out, K, L_tilde):
        super().__init__()
        self.K = K
        self.register_buffer("L_tilde",
                             torch.tensor(L_tilde, dtype=torch.float32))
        self.weight = nn.Parameter(torch.FloatTensor(K, c_in, c_out))
        self.bias   = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        B, T, N, C = x.shape
        out    = torch.zeros(B, T, N, self.weight.shape[-1], device=x.device)
        x_flat = x.reshape(B * T, N, C)
        L      = self.L_tilde.unsqueeze(0).expand(B * T, -1, -1)
        Tx_0   = x_flat
        Tx_1   = torch.bmm(L, x_flat)
        out_f  = torch.einsum('bnc,co->bno', Tx_0, self.weight[0])
        if self.K > 1:
            out_f = out_f + torch.einsum('bnc,co->bno', Tx_1, self.weight[1])
        for k in range(2, self.K):
            Tx_2  = 2 * torch.bmm(L, Tx_1) - Tx_0
            out_f = out_f + torch.einsum('bnc,co->bno', Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        out_f = out_f + self.bias
        return out_f.reshape(B, T, N, -1)


class TemporalConv(nn.Module):
    def __init__(self, c_in, c_out, Kt):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(Kt, 1))
        self.bn   = nn.BatchNorm2d(2 * c_out)

    def forward(self, x):
        x   = x.permute(0, 3, 1, 2)
        out = self.bn(self.conv(x))
        P, Q = out.chunk(2, dim=1)
        return (P * torch.sigmoid(Q)).permute(0, 2, 3, 1)


class STConvBlock(nn.Module):
    def __init__(self, c_in, c_spatial, c_out, Kt, K, L_tilde):
        super().__init__()
        self.t1    = TemporalConv(c_in,      c_out,     Kt)
        self.gc    = ChebConv(c_out,         c_spatial, K, L_tilde)
        self.gc_bn = nn.BatchNorm1d(c_spatial)   # must match training checkpoint
        self.t2    = TemporalConv(c_spatial, c_out,     Kt)
        self.norm  = nn.LayerNorm(c_out)
        self.res   = nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.1)

    def forward(self, x):
        T_in = x.shape[1]
        res  = x
        out  = self.t1(x)
        gc   = self.gc(out)
        gc   = self.relu(gc)
        gc   = self.drop(gc)
        out  = self.t2(gc)
        T_out = out.shape[1]
        trim  = T_in - T_out
        res   = res[:, trim // 2 : T_in - (trim - trim // 2), :, :]
        res   = self.res(res)
        return self.norm(out + res)


class STGCN(nn.Module):
    def __init__(self, C, N, T_in, T_out, K, Kt, L_tilde, c_s=16, c_out=64):
        super().__init__()
        self.t_out_steps = T_out
        self.block1  = STConvBlock(C,     c_s, c_out, Kt, K, L_tilde)
        self.block2  = STConvBlock(c_out, c_s, c_out, Kt, K, L_tilde)
        self.t_out   = TemporalConv(c_out, c_out, Kt)
        self.fc      = nn.Linear(c_out, C)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.block1(x)
        out = self.dropout(out)
        out = self.block2(out)
        out = self.t_out(out)
        out = self.fc(out)
        out = out[:, -self.t_out_steps:, :, :]
        return out


# ══════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════

def load_model(model_path, tensor_path, laplacian_path):
    checkpoint  = torch.load(model_path, map_location=DEVICE)
    cfg         = checkpoint["config"]
    T, N, C     = checkpoint["T"], checkpoint["N"], checkpoint["C"]

    L_df    = pd.read_csv(laplacian_path, index_col=0)
    L_tilde = L_df.values.astype(np.float32)
    lam_max = np.linalg.eigvalsh(L_tilde).max()
    L_tilde = (2.0 / (lam_max + 1e-8)) * L_tilde - np.eye(N, dtype=np.float32)

    model = STGCN(
        C=C, N=N,
        T_in=cfg["t_in"], T_out=cfg["t_out"],
        K=cfg["k"], Kt=cfg["kt"],
        L_tilde=L_tilde,
        c_s=cfg["c_spatial"], c_out=cfg["c_out"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"  Model loaded: T={T}, N={N}, C={C}")
    print(f"  Config: T_in={cfg['t_in']}, K={cfg['k']}, Kt={cfg['kt']}, "
          f"c_out={cfg['c_out']}")
    return model, cfg, N, C


def load_normalisation(stgcn_dir):
    mean_path = Path(stgcn_dir) / "feat_mean.npy"
    std_path  = Path(stgcn_dir) / "feat_std.npy"
    if mean_path.exists() and std_path.exists():
        feat_mean = np.load(mean_path)
        feat_std  = np.load(std_path)
        print(f"  Normalisation loaded: mean={feat_mean.round(3)}")
        return feat_mean, feat_std
    print("  WARNING: feat_mean/feat_std not found — predictions stay z-scored")
    return None, None


# ══════════════════════════════════════════════════════════════════════════
# AUTOREGRESSIVE FORECAST
# ══════════════════════════════════════════════════════════════════════════

def forecast(model, X_norm, t_in, n_steps):
    """
    Autoregressively predicts n_steps beyond the end of X_norm.
    Seeds with the last t_in real observations.
    Returns predictions: [n_steps, N, C]
    """
    model.eval()
    # Seed window: last t_in time steps
    window = X_norm[-t_in:].copy()          # [t_in, N, C]
    preds  = []

    with torch.no_grad():
        for step in range(n_steps):
            inp = torch.tensor(
                window[np.newaxis],          # [1, t_in, N, C]
                dtype=torch.float32, device=DEVICE
            )
            out = model(inp)                 # [1, 1, N, C]
            pred_step = out[0, 0].cpu().numpy()   # [N, C]
            preds.append(pred_step)

            # Slide window: drop oldest, append prediction
            window = np.concatenate(
                [window[1:], pred_step[np.newaxis]], axis=0
            )
            print(f"  Step +{step+1}/{n_steps} done")

    return np.array(preds)   # [n_steps, N, C]


def denormalise(preds_norm, feat_mean, feat_std):
    """Convert z-scored predictions back to original scale."""
    if feat_mean is None:
        return preds_norm
    # preds_norm: [n_steps, N, C]
    # feat_mean/std: [C]
    return preds_norm * feat_std[np.newaxis, np.newaxis, :] \
           + feat_mean[np.newaxis, np.newaxis, :]


# ══════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════

def plot_national_forecast(X_orig, preds, feat_mean, feat_std, n_steps):
    """
    National total enrolment: last 20 real steps + n_steps forecast.
    """
    # Denormalise X_orig for the real history
    if feat_mean is not None:
        X_real = X_orig * feat_std[np.newaxis, np.newaxis, :] \
                 + feat_mean[np.newaxis, np.newaxis, :]
    else:
        X_real = X_orig

    # National total (C=0 = enrol_total)
    real_nat = X_real[-20:, :, 0].sum(axis=1)    # [20]
    pred_nat = preds[:, :, 0].sum(axis=1)         # [n_steps]

    x_real = np.arange(len(real_nat))
    x_pred = np.arange(len(real_nat) - 1, len(real_nat) + n_steps)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_facecolor("#ffffff")

    ax.plot(x_real, real_nat, color="#534AB7", linewidth=2.5,
            label="Historical (last 20 steps)")
    ax.plot(x_pred, np.append(real_nat[-1], pred_nat),
            color="#D85A30", linewidth=2.5, linestyle="--",
            label=f"Forecast (+{n_steps} steps)")
    ax.fill_between(x_pred,
                    np.append(real_nat[-1], pred_nat) * 0.92,
                    np.append(real_nat[-1], pred_nat) * 1.08,
                    color="#D85A30", alpha=0.12,
                    label="±8% uncertainty band")

    ax.axvline(len(real_nat) - 1, color="#9ca3af",
               linewidth=1, linestyle="--", alpha=0.6)
    ax.text(len(real_nat) - 0.6, ax.get_ylim()[1] * 0.98,
            "Forecast →", fontsize=9, color="#9ca3af")

    ax.set_xlabel("Time step", fontsize=10)
    ax.set_ylabel("National enrolment total", fontsize=10)
    ax.set_title("STGCN forecast — national Aadhaar enrolment",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.legend(fontsize=9)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color("#e5e7eb")
    ax.grid(True, axis="y", color="#f0f0f0")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_national.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → forecast_national.png")


def plot_top_states(preds, districts_df, n_steps):
    """
    Top 10 states by total forecast enrolment — line chart per state.
    """
    # Aggregate districts to states using district_order.csv
    # district_order has columns: index, district
    dist_list = districts_df.iloc[:, 1].tolist() \
                if districts_df.shape[1] > 1 \
                else districts_df.iloc[:, 0].tolist()

    # preds: [n_steps, N, C=0 = enrol_total]
    enrol_pred = preds[:, :, 0]   # [n_steps, N]

    # We don't have state mapping in forecast script —
    # show top 10 districts by total forecast instead
    total_per_dist = enrol_pred.sum(axis=0)  # [N]
    top10_idx = np.argsort(total_per_dist)[-10:][::-1]

    colors = ["#1D9E75","#534AB7","#D85A30","#BA7517","#D4537E",
              "#378ADD","#639922","#E24B4A","#5DCAA5","#7F77DD"]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor("#ffffff")

    for i, di in enumerate(top10_idx):
        name = dist_list[di][:18] if di < len(dist_list) else f"District {di}"
        ax.plot(range(n_steps), enrol_pred[:, di],
                color=colors[i % len(colors)], linewidth=2,
                label=name, alpha=0.85)
        ax.scatter(range(n_steps), enrol_pred[:, di],
                   color=colors[i % len(colors)], s=25, zorder=3, alpha=0.7)

    ax.set_xlabel("Forecast step", fontsize=10)
    ax.set_ylabel("Predicted enrolment (z-scored)" \
                  if True else "Predicted enrolment", fontsize=10)
    ax.set_title("STGCN forecast — top 10 districts by enrolment",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f"+{i+1}" for i in range(n_steps)])
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color("#e5e7eb")
    ax.grid(True, axis="y", color="#f0f0f0")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_top_districts.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → forecast_top_districts.png")


def plot_feature_forecasts(preds, n_steps):
    """
    Small multiples: national forecast for each of the 7 features.
    """
    national = preds.sum(axis=1)   # [n_steps, C]
    C        = national.shape[1]
    n_feats  = min(C, len(FEATURE_NAMES))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor=BG)
    axes = axes.flatten()

    colors = ["#1D9E75","#534AB7","#D85A30","#BA7517",
              "#D4537E","#378ADD","#9ca3af"]

    for i in range(n_feats):
        ax = axes[i]
        ax.set_facecolor("#ffffff")
        ax.plot(range(n_steps), national[:, i],
                color=colors[i], linewidth=2.2)
        ax.scatter(range(n_steps), national[:, i],
                   color=colors[i], s=30, zorder=3)
        ax.fill_between(range(n_steps),
                        national[:, i] * 0.92,
                        national[:, i] * 1.08,
                        color=colors[i], alpha=0.1)
        ax.set_title(FEATURE_NAMES[i][:22], fontsize=9, fontweight="bold")
        ax.set_xticks(range(n_steps))
        ax.set_xticklabels([f"+{j+1}" for j in range(n_steps)], fontsize=7)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax.spines[sp].set_color("#e5e7eb")
        ax.grid(True, axis="y", color="#f0f0f0", linewidth=0.6)

    # Hide unused subplot
    for i in range(n_feats, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("STGCN forecast — national totals per feature",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_all_features.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → forecast_all_features.png")


def plot_change_heatmap(preds, districts_df, n_steps):
    """
    Shows % change from step+1 to step+N for top 30 districts.
    """
    dist_list = districts_df.iloc[:, 1].tolist() \
                if districts_df.shape[1] > 1 \
                else districts_df.iloc[:, 0].tolist()

    enrol = preds[:, :, 0]   # [n_steps, N]
    base  = enrol[0]          # step+1 as baseline

    # Avoid divide by zero
    safe_base = np.where(np.abs(base) > 1e-8, base, 1e-8)
    pct_change = (enrol - base[np.newaxis, :]) / np.abs(safe_base[np.newaxis, :]) * 100

    # Top 30 districts by absolute forecast enrolment
    top30_idx  = np.argsort(enrol.mean(axis=0))[-30:][::-1]
    top30_pct  = pct_change[:, top30_idx].T   # [30, n_steps]
    top30_names = [dist_list[i][:16] if i < len(dist_list)
                   else f"D{i}" for i in top30_idx]

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 9), facecolor=BG)
    sns.heatmap(
        top30_pct, ax=ax,
        cmap="RdYlGn", center=0,
        xticklabels=[f"+{i+1}" for i in range(n_steps)],
        yticklabels=top30_names,
        annot=True, fmt=".1f",
        annot_kws={"size": 7},
        linewidths=0.3,
        linecolor="#f0f0f0",
        cbar_kws={"shrink": 0.5, "label": "% change from step+1"},
    )
    ax.set_title("Forecast % change — top 30 districts",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_change_heatmap.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → forecast_change_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# SAVE DATA
# ══════════════════════════════════════════════════════════════════════════

def save_data(preds_denorm, preds_norm, districts_df, n_steps):
    """Save forecast as CSV — one row per district per step."""
    dist_list = districts_df.iloc[:, 1].tolist() \
                if districts_df.shape[1] > 1 \
                else districts_df.iloc[:, 0].tolist()
    N = preds_denorm.shape[1]

    rows = []
    for step in range(n_steps):
        for di in range(N):
            row = {"step": f"+{step+1}", "district": dist_list[di] if di < len(dist_list) else f"District_{di}"}
            for ci, fname in enumerate(FEATURE_NAMES[:preds_denorm.shape[2]]):
                row[fname] = round(float(preds_denorm[step, di, ci]), 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "forecast.csv", index=False)
    print(f"  Saved → forecast.csv  ({len(df):,} rows)")

    # National summary
    nat_rows = []
    for step in range(n_steps):
        row = {"step": f"+{step+1}"}
        for ci, fname in enumerate(FEATURE_NAMES[:preds_denorm.shape[2]]):
            row[f"{fname}_national_total"] = float(preds_denorm[step, :, ci].sum())
        nat_rows.append(row)
    pd.DataFrame(nat_rows).to_csv(OUTPUT_DIR / "forecast_summary.csv", index=False)
    print("  Saved → forecast_summary.csv")

    np.save(OUTPUT_DIR / "forecast_raw.npy", preds_norm)
    print("  Saved → forecast_raw.npy")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 60)
    print("STGCN FUTURE FORECAST")
    print("=" * 60)
    print(f"  Forecast steps: {args.steps}")
    print(f"  Model:          {args.model}")

    # Load model
    print("\n── Loading model ──")
    model, cfg, N, C = load_model(
        args.model, args.tensor, args.laplacian)
    t_in = cfg["t_in"]

    # Load normalisation
    print("\n── Loading normalisation ──")
    feat_mean, feat_std = load_normalisation(args.stgcn_dir)

    # Load tensor and apply z-score normalisation
    print("\n── Loading tensor ──")
    X = np.load(args.tensor).astype(np.float32)
    print(f"  Tensor shape: {X.shape}")

    # Apply same z-score normalisation used during training
    if feat_mean is not None:
        X_norm = (X - feat_mean[np.newaxis, np.newaxis, :]) \
                 / (feat_std[np.newaxis, np.newaxis, :] + 1e-8)
    else:
        # Fallback: normalise on the fly
        mean = X.mean(axis=(0, 1), keepdims=True)
        std  = X.std(axis=(0, 1),  keepdims=True)
        std[std < 1e-8] = 1.0
        X_norm    = (X - mean) / std
        feat_mean = mean.squeeze()
        feat_std  = std.squeeze()

    # Load district names
    districts_df = pd.read_csv(args.districts)

    # Autoregressive forecast
    print(f"\n── Forecasting {args.steps} steps ahead ──")
    preds_norm = forecast(model, X_norm, t_in, args.steps)
    print(f"  Forecast shape: {preds_norm.shape}")

    # Denormalise
    preds_denorm = denormalise(preds_norm, feat_mean, feat_std)

    # Charts
    print("\n── Generating charts ──")
    plot_national_forecast(X_norm, preds_norm, feat_mean, feat_std, args.steps)
    plot_top_states(preds_norm, districts_df, args.steps)
    plot_feature_forecasts(preds_norm, args.steps)
    plot_change_heatmap(preds_norm, districts_df, args.steps)

    # Save data
    print("\n── Saving data ──")
    save_data(preds_denorm, preds_norm, districts_df, args.steps)

    print("\n" + "=" * 60)
    print("DONE — forecast_output/")
    print("  forecast_national.png       national trend + forecast")
    print("  forecast_top_districts.png  top 10 districts forecast")
    print("  forecast_all_features.png   all 7 features forecast")
    print("  forecast_change_heatmap.png % change heatmap")
    print("  forecast.csv                per-district per-step values")
    print("  forecast_summary.csv        national step aggregates")
    print("  forecast_raw.npy            raw z-scored predictions")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="STGCN autoregressive forecast")
    p.add_argument("--steps",      type=int,   default=6,
                   help="Number of future steps to forecast")
    p.add_argument("--model",      default="../STCGN/stgcn_output/best_model.pt")
    p.add_argument("--tensor",     default="../STCGN/adjacency_output/feature_tensor_X.npy")
    p.add_argument("--laplacian",  default="../STCGN/adjacency_output/L_normalised_laplacian.csv")
    p.add_argument("--districts",  default="../STCGN/adjacency_output/district_order.csv")
    p.add_argument("--stgcn_dir",  default="../STCGN/stgcn_output",
                   help="Folder containing feat_mean.npy and feat_std.npy")
    a = p.parse_args()
    main(a)