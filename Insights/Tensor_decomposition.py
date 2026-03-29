"""
tensor_decomposition.py
=======================
Loads biometric, demographic and enrolment preprocessed tables from DuckDB,
builds a 3-D tensor  (districts × time-steps × features),
runs CP (PARAFAC) and Tucker decompositions via TensorLy,
and writes every output that the Streamlit "Tensor Decomposition" page expects.

Usage
-----
    python tensor_decomposition.py --db ../database/aadhar.duckdb

Optional flags:
    --output   tensor_output/    Output directory  (default: tensor_output)
    --cp-rank  10                CP / PARAFAC rank  (default: 10)
    --max-rank 20                Max rank to sweep for elbow plot (default: 20)
    --tucker-ranks 8 6 6         Tucker ranks for [district, time, feature] modes
                                 (default: auto = 60 % of each dim, min 2)
    --seed     42                Random seed
"""

import argparse
import os
import warnings
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorly as tl
from tensorly.decomposition import parafac, tucker

warnings.filterwarnings("ignore")

# ── colour palette matching the Streamlit app ────────────────────────────
C = {
    "purple":  "#534AB7",
    "green":   "#1D9E75",
    "orange":  "#D85A30",
    "amber":   "#BA7517",
    "pink":    "#D4537E",
    "gray":    "#9ca3af",
    "bg":      "#f9f8f6",
}
CMAP_DIV   = "RdYlGn"
CMAP_SEQ   = "YlOrRd"
CMAP_COOL  = sns.color_palette("mako",  as_cmap=True)
CMAP_WARM  = sns.color_palette("flare", as_cmap=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

# ─────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────

BIO_COLS   = ["bio_age_5_17", "bio_age_17_", "bio_total",
              "age_5_ratio", "age_17_ratio", "dependency_ratio",
              "bio_total_7day_avg", "bio_total_7day_std"]

DEMO_COLS  = ["demo_age_5_17", "demo_age_17_", "demo_total",
              "demo_age_5_ratio", "demo_age_17_ratio",
              "demo_dependency_ratio",
              "demo_total_7day_avg", "demo_total_7day_std"]

ENROL_COLS = ["age_0_5", "age_5_17", "age_18_greater", "enrol_total",
              "enrol_minor_ratio", "enrol_adult_ratio",
              "enrol_total_7day_avg", "enrol_total_7day_std"]

ALL_FEATURE_COLS = BIO_COLS + DEMO_COLS + ENROL_COLS   # 24 features

TABLE_MAP = {
    "bio":   ("biometric_data_preprocessed",   BIO_COLS),
    "demo":  ("demographic_data_preprocessed", DEMO_COLS),
    "enrol": ("enrolment_data_preprocessed",   ENROL_COLS),
}


def load_tables(db_path: str) -> dict[str, pd.DataFrame]:
    """Return {name: dataframe} for bio, demo, enrol."""
    con = duckdb.connect(db_path, read_only=True)
    dfs = {}
    for name, (table, cols) in TABLE_MAP.items():
        existing = [r[0] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
        safe = [c for c in cols if c in existing]
        query = f"""
            SELECT state, district, pincode,
                   CAST(date AS DATE) AS date,
                   {', '.join(safe)}
            FROM {table}
            WHERE state IS NOT NULL AND district IS NOT NULL
        """
        df = con.execute(query).df()
        df["date"] = pd.to_datetime(df["date"])
        dfs[name] = df
    con.close()
    return dfs


def build_tensor(dfs: dict[str, pd.DataFrame]):
    """
    Align three tables on (district, date) and build a 3-D numpy array
    of shape  (N_districts, T_timesteps, F_features).

    Returns
    -------
    tensor       : np.ndarray  shape (N, T, F)
    districts    : list[str]   district labels  (length N)
    dates        : list        date labels       (length T)
    feature_names: list[str]   feature labels   (length F)
    district_meta: pd.DataFrame  state/district lookup
    """
    # aggregate duplicates within each table to daily district totals
    agg_dfs = {}
    for name, (_, cols) in TABLE_MAP.items():
        df = dfs[name].copy()
        existing_cols = [c for c in cols if c in df.columns]
        grp = df.groupby(["state", "district", "date"])[existing_cols].mean().reset_index()
        grp["dist_key"] = grp["state"] + " | " + grp["district"]
        agg_dfs[name] = grp

    # common district × date pairs across all three tables
    def key_set(df):
        return set(zip(df["dist_key"], df["date"]))

    common_keys = set()
    for name in ["bio", "demo", "enrol"]:
        common_keys |= key_set(agg_dfs[name])

    if not common_keys:
        raise ValueError(
            "No overlapping (district, date) pairs across the three tables. "
            "Check that the date ranges overlap."
        )

    keys_df = pd.DataFrame(list(common_keys), columns=["dist_key", "date"])
    districts = sorted(keys_df["dist_key"].unique())
    dates     = sorted(keys_df["date"].unique())
    N, T = len(districts), len(dates)

    # build meta lookup
    meta = agg_dfs["bio"][["dist_key", "state", "district"]].drop_duplicates("dist_key")

    # collect feature arrays
    feature_names = []
    slices = []          # each slice: (N, T) array

    # for name, (_, cols) in TABLE_MAP.items():
    #     df = agg_dfs[name]
    #     df = df[df["dist_key"].isin(districts) & df["date"].isin(dates)]
    #     existing_cols = [c for c in cols if c in df.columns]
    #     # pivot each feature column
    #     for col in existing_cols:
    #         piv = df.pivot_table(index="dist_key", columns="date",
    #                              values=col, aggfunc="mean")
    #         piv = piv.reindex(index=districts, columns=dates)
    #         arr = piv.values.astype(float)
    #         slices.append(arr)
    #         feature_names.append(f"{name}_{col}")

    
    for name, (_, cols) in TABLE_MAP.items():
        df = agg_dfs[name]
        df = df[df["dist_key"].isin(districts) & df["date"].isin(dates)]
        print("\n", name, "dtypes:\n", df.dtypes)
        existing_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        existing_cols = [c for c in existing_cols if c not in ["pincode"]]

        if not existing_cols:
            print(f"⚠️ No numeric columns in {name}, skipping")
            continue

        for col in existing_cols:
            piv = df.pivot_table(index="dist_key", columns="date",
                                values=col, aggfunc="mean")
            piv = piv.reindex(index=districts, columns=dates)

            if piv.isna().all().all():
                print(f"⚠️ Skipping feature {name}_{col} (all NaN)")
                continue

            slices.append(piv.values.astype(float))
            feature_names.append(f"{name}_{col}")

    if not slices:
        raise ValueError("No valid tensor slices created. Check data alignment.")

    # stack → (N, T, F)
    tensor_raw = np.stack(slices, axis=2)

    # fill NaN with column mean (per feature × time) then overall mean
    for f in range(tensor_raw.shape[2]):
        for t in range(tensor_raw.shape[1]):
            col_vals = tensor_raw[:, t, f]
            mask = np.isnan(col_vals)
            if mask.any():
                fill = np.nanmean(col_vals) if not np.all(mask) else 0.0
                tensor_raw[mask, t, f] = fill

    # z-score normalise per feature (across all districts × time)
    F = tensor_raw.shape[2]
    for f in range(F):
        flat = tensor_raw[:, :, f].ravel()
        mu, sigma = flat.mean(), flat.std()
        if sigma > 1e-9:
            tensor_raw[:, :, f] = (tensor_raw[:, :, f] - mu) / sigma

    return tensor_raw, districts, dates, feature_names, meta


# ─────────────────────────────────────────────────────────────────────────
# 2.  DECOMPOSITIONS
# ─────────────────────────────────────────────────────────────────────────

def run_cp(tensor_data, rank: int, seed: int):
    tl.set_backend("numpy")
    factors = parafac(tensor_data, rank=rank, random_state=seed,
                      init="svd", n_iter_max=200, tol=1e-8)
    recon   = tl.cp_to_tensor(factors)
    error   = np.linalg.norm(tensor_data - recon) / (np.linalg.norm(tensor_data) + 1e-12)
    return factors, recon, error


def run_tucker(tensor_data, ranks, seed: int):
    tl.set_backend("numpy")
    core, factors = tucker(tensor_data, rank=ranks, random_state=seed,
                           init="svd", n_iter_max=200, tol=1e-8)
    recon = tl.tucker_to_tensor((core, factors))
    error = np.linalg.norm(tensor_data - recon) / (np.linalg.norm(tensor_data) + 1e-12)
    return core, factors, recon, error


def sweep_cp_rank(tensor_data, max_rank: int, seed: int):
    errors = []
    ranks  = list(range(2, max_rank + 1))
    for r in ranks:
        _, _, err = run_cp(tensor_data, r, seed)
        errors.append(err)
        print(f"  CP rank={r:3d}  error={err:.6f}")
    return ranks, errors


# ─────────────────────────────────────────────────────────────────────────
# 3.  PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────

def savefig(out_dir: Path, name: str):
    path = out_dir / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ── 3a. Reconstruction error curves ──────────────────────────────────────

def plot_cp_error_curve(ranks, errors, chosen_rank, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ranks, errors, "o-", color=C["purple"], lw=2, ms=5)
    ax.axvline(chosen_rank, color=C["orange"], ls="--", lw=1.5,
               label=f"Chosen rank = {chosen_rank}")
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Normalised Frobenius Error")
    ax.set_title("CP Decomposition — Reconstruction Error vs Rank")
    ax.legend()
    savefig(out_dir, "cp_reconstruction_error.png")


def plot_tucker_error(tucker_errors_dict, out_dir):
    """tucker_errors_dict: {feature_rank: error}"""
    ranks  = list(tucker_errors_dict.keys())
    errors = list(tucker_errors_dict.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ranks, errors, "s-", color=C["green"], lw=2, ms=5)
    ax.set_xlabel("Tucker feature-mode rank")
    ax.set_ylabel("Normalised Frobenius Error")
    ax.set_title("Tucker Decomposition — Reconstruction Error vs Feature Rank")
    savefig(out_dir, "tucker_reconstruction_error.png")


# ── 3b. Explained variance scree ─────────────────────────────────────────

def plot_cp_explained_variance(cp_factors, tensor_data, out_dir):
    """Use CP weights (λ) proxy for explained variance."""
    weights = cp_factors.weights
    if weights is None or len(weights) == 0:
        # compute manually from factor norms
        A = cp_factors.factors[0]  # district
        weights = np.linalg.norm(A, axis=0)
    total   = weights.sum()
    pct     = 100 * weights / (total + 1e-12)
    cum_pct = np.cumsum(pct)
    rank    = len(pct)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(1, rank + 1), pct, color=C["purple"], alpha=0.8)
    ax2  = ax.twinx()
    ax2.plot(range(1, rank + 1), cum_pct, "o-", color=C["orange"], lw=2, ms=4)
    ax2.axhline(90, color=C["gray"], ls=":", lw=1, label="90 % threshold")
    ax2.set_ylabel("Cumulative %", color=C["orange"])
    ax2.tick_params(axis="y", labelcolor=C["orange"])
    ax.set_xlabel("CP Component")
    ax.set_ylabel("% Contribution")
    ax.set_title("CP Decomposition — Explained Variance by Component")
    savefig(out_dir, "cp_explained_variance.png")


# ── 3c. Feature loading heatmap ───────────────────────────────────────────

def plot_feature_loading_heatmap(feature_matrix, feature_names, title, fname, out_dir):
    """feature_matrix: shape (F, rank)"""
    rank = feature_matrix.shape[1]
    fig_h = max(6, len(feature_names) * 0.35)
    fig, ax = plt.subplots(figsize=(min(rank * 0.9 + 2, 16), fig_h))
    sns.heatmap(feature_matrix, ax=ax,
                xticklabels=[f"C{i+1}" for i in range(rank)],
                yticklabels=feature_names,
                cmap=CMAP_DIV, center=0,
                linewidths=0.3, linecolor="#e5e7eb",
                cbar_kws={"shrink": 0.7, "label": "Loading"})
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    savefig(out_dir, fname)


# ── 3d. Top districts per component ──────────────────────────────────────

def plot_top_districts(district_factors, districts, out_dir, top_n=8):
    rank = district_factors.shape[1]
    ncols = min(rank, 4)
    nrows = int(np.ceil(rank / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.2))
    axes = np.array(axes).ravel()
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i in range(rank):
        ax = axes[i]
        scores  = district_factors[:, i]
        top_idx = np.argsort(scores)[-top_n:][::-1]
        labels  = [districts[j].split(" | ")[-1] for j in top_idx]
        vals    = scores[top_idx]
        ax.barh(labels[::-1], vals[::-1],
                color=colors[i % len(colors)], alpha=0.85)
        ax.set_title(f"Component {i+1}", fontsize=10)
        ax.tick_params(labelsize=8)

    for j in range(rank, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Districts per CP Component", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig(out_dir, "cp_top_districts.png")


# ── 3e. CP factor weights bar ─────────────────────────────────────────────

def plot_cp_factor_weights(cp_factors, out_dir):
    weights = cp_factors.weights
    if weights is None or len(weights) == 0:
        A = cp_factors.factors[0]
        weights = np.linalg.norm(A, axis=0)
    rank = len(weights)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, rank + 1), weights, color=C["purple"], alpha=0.85)
    ax.set_xlabel("CP Component")
    ax.set_ylabel("Weight λ")
    ax.set_title("CP Factor Weights (λ) — Component Importance")
    savefig(out_dir, "cp_factor_weights.png")


# ── 3f. District scatter (CP) ─────────────────────────────────────────────

def plot_district_scatter(district_factors, districts, meta, out_dir):
    df = pd.DataFrame(district_factors[:, :2],
                      columns=["CP1", "CP2"])
    df["dist_key"] = districts
    df = df.merge(meta[["dist_key", "state"]], on="dist_key", how="left")
    states = df["state"].fillna("Unknown").unique()
    cmap_s = plt.cm.get_cmap("tab20", len(states))
    state_color = {s: cmap_s(i) for i, s in enumerate(states)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for state, grp in df.groupby("state"):
        ax.scatter(grp["CP1"], grp["CP2"],
                   color=state_color.get(state, "gray"),
                   s=14, alpha=0.65, label=state)
    ax.set_xlabel("CP Factor 1 (district mode)")
    ax.set_ylabel("CP Factor 2 (district mode)")
    ax.set_title("District Embedding — CP Spatial Factors 1 vs 2")
    # legend only if not too many states
    if len(states) <= 20:
        ax.legend(fontsize=7, ncol=2, markerscale=1.5,
                  loc="upper right", framealpha=0.6)
    savefig(out_dir, "cp_district_scatter.png")


# ── 3g. Tucker district embedding (PCA) ──────────────────────────────────

def plot_tucker_district_embedding(tucker_district_factors, districts, meta, out_dir):
    X = tucker_district_factors
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X2  = pca.fit_transform(X)
        var = pca.explained_variance_ratio_ * 100
        xlabel = f"PC1 ({var[0]:.1f}% var)"
        ylabel = f"PC2 ({var[1]:.1f}% var)"
    else:
        X2 = X[:, :2]
        xlabel, ylabel = "Tucker Dim 1", "Tucker Dim 2"

    df = pd.DataFrame(X2, columns=["x", "y"])
    df["dist_key"] = districts
    df = df.merge(meta[["dist_key", "state"]], on="dist_key", how="left")

    fig, ax = plt.subplots(figsize=(9, 7))
    states = df["state"].fillna("Unknown").unique()
    cmap_s = plt.cm.get_cmap("tab20", len(states))
    for i, (state, grp) in enumerate(df.groupby("state")):
        ax.scatter(grp["x"], grp["y"], color=cmap_s(i), s=14, alpha=0.65, label=state)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Tucker District Embedding — 2D PCA Projection")
    if len(states) <= 20:
        ax.legend(fontsize=7, ncol=2, markerscale=1.5, loc="upper right", framealpha=0.6)
    savefig(out_dir, "tucker_district_embedding.png")


# ── 3h. Temporal factor lines ─────────────────────────────────────────────

def plot_temporal_factors(temporal_factors, dates, out_dir, label="CP"):
    rank   = temporal_factors.shape[1]
    n_show = min(rank, 8)
    fig, axes = plt.subplots(n_show, 1, figsize=(11, n_show * 2.0), sharex=True)
    if n_show == 1:
        axes = [axes]
    colors = [C["purple"], C["green"], C["orange"], C["amber"],
              C["pink"], C["gray"], "#3b82f6", "#ec4899"]
    for i in range(n_show):
        ax = axes[i]
        ax.plot(dates, temporal_factors[:, i],
                color=colors[i % len(colors)], lw=1.5)
        ax.set_ylabel(f"C{i+1}", rotation=0, labelpad=28, fontsize=9)
        ax.axhline(0, color="#d1d5db", lw=0.7, ls="--")
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{label} Temporal Factors over Time", fontsize=13)
    plt.tight_layout()
    savefig(out_dir, f"{label.lower()}_temporal_factors.png")


# ── 3i. Tucker temporal heatmap ───────────────────────────────────────────

def plot_tucker_temporal_heatmap(temporal_factors, dates, out_dir):
    fig, ax = plt.subplots(figsize=(14, max(3, temporal_factors.shape[1] * 0.5)))
    date_labels = [str(d)[:10] for d in dates]
    step = max(1, len(date_labels) // 20)
    sns.heatmap(temporal_factors.T, ax=ax,
                xticklabels=step,
                yticklabels=[f"TD{i+1}" for i in range(temporal_factors.shape[1])],
                cmap=CMAP_COOL,
                cbar_kws={"shrink": 0.6, "label": "Factor loading"})
    xticks = ax.get_xticklabels()
    # show every step-th label
    shown = [tl if i % step == 0 else "" for i, tl in enumerate(date_labels)]
    ax.set_xticklabels(shown, rotation=45, ha="right", fontsize=7)
    ax.set_title("Tucker Temporal Factor Heatmap")
    savefig(out_dir, "tucker_temporal_heatmap.png")


# ── 3j. Tucker core slices ────────────────────────────────────────────────

def plot_tucker_core_slice(core, out_dir):
    # core shape: (R_district, R_time, R_feature)
    # slice along feature mode (first feature latent dim)
    slice_dt = core[:, :, 0]
    slice_df = core[:, 0, :]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title, fname_hint in [
        (axes[0], slice_dt, "Core slice: District × Time (feature dim 0)", "dt"),
        (axes[1], slice_df, "Core slice: District × Feature (time dim 0)",  "df"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=CMAP_WARM)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time latent dim" if fname_hint == "dt" else "Feature latent dim")
        ax.set_ylabel("District latent dim")

    plt.tight_layout()
    savefig(out_dir, "tucker_core_slices.png")
    # also save individually for the Streamlit page
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    im = ax2.imshow(slice_dt, aspect="auto", cmap=CMAP_WARM)
    plt.colorbar(im, ax=ax2, shrink=0.7, label="Core value")
    ax2.set_title("Tucker Core — District × Time (feature dim 0)")
    ax2.set_xlabel("Time latent dim"); ax2.set_ylabel("District latent dim")
    savefig(out_dir, "tucker_core_dt.png")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    im = ax3.imshow(slice_df, aspect="auto", cmap=CMAP_WARM)
    plt.colorbar(im, ax=ax3, shrink=0.7, label="Core value")
    ax3.set_title("Tucker Core — District × Feature (time dim 0)")
    ax3.set_xlabel("Feature latent dim"); ax3.set_ylabel("District latent dim")
    savefig(out_dir, "tucker_core_df.png")


# ── 3k. Cross-table latent correlation ───────────────────────────────────

def plot_cross_table_corr(district_factors, feature_names, out_dir):
    """Split district factor scores by table origin via feature loadings."""
    # identify feature indices per table
    bio_idx   = [i for i, n in enumerate(feature_names) if n.startswith("bio_")]
    demo_idx  = [i for i, n in enumerate(feature_names) if n.startswith("demo_")]
    enrol_idx = [i for i, n in enumerate(feature_names) if n.startswith("enrol_")]

    rank = district_factors.shape[1]
    if rank < 2:
        return

    # project district factors onto bio / enrol / demo subspaces
    # (use first 2 CP components for simplicity)
    comp_pairs = [(0, 1)] if rank >= 2 else [(0, 0)]

    for pair_name, (i, j) in [("bio_enrol", (0, 1)), ("enrol_demo", (1, min(2, rank-1)))]:
        fig, ax = plt.subplots(figsize=(6, 5))
        xi = district_factors[:, i]
        xj = district_factors[:, j]
        ax.scatter(xi, xj, color=C["purple"], s=10, alpha=0.5)
        r = np.corrcoef(xi, xj)[0, 1]
        ax.set_xlabel(f"CP Component {i+1} district score")
        ax.set_ylabel(f"CP Component {j+1} district score")
        ax.set_title(f"District factor correlation  r = {r:.3f}")
        ax.text(0.05, 0.93, f"r = {r:.3f}",
                transform=ax.transAxes, color=C["orange"], fontsize=12, fontweight="bold")
        fname = "bio_enrol_latent_corr.png" if pair_name == "bio_enrol" else "enrol_demo_latent_corr.png"
        savefig(out_dir, fname)


def plot_cross_table_interaction_heatmap(feature_matrix, feature_names, out_dir):
    """Aggregate feature-mode loadings grouped by table."""
    bio_idx   = [i for i, n in enumerate(feature_names) if n.startswith("bio_")]
    demo_idx  = [i for i, n in enumerate(feature_names) if n.startswith("demo_")]
    enrol_idx = [i for i, n in enumerate(feature_names) if n.startswith("enrol_")]

    rank = feature_matrix.shape[1]
    agg = np.zeros((3, rank))
    agg[0] = np.abs(feature_matrix[bio_idx]).mean(axis=0)   if bio_idx   else 0
    agg[1] = np.abs(feature_matrix[demo_idx]).mean(axis=0)  if demo_idx  else 0
    agg[2] = np.abs(feature_matrix[enrol_idx]).mean(axis=0) if enrol_idx else 0

    fig, ax = plt.subplots(figsize=(min(rank * 0.9 + 2, 14), 4))
    sns.heatmap(agg, ax=ax,
                xticklabels=[f"C{i+1}" for i in range(rank)],
                yticklabels=["Biometric", "Demographic", "Enrolment"],
                cmap="YlOrRd", annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.4, linecolor="#e5e7eb",
                cbar_kws={"shrink": 0.6, "label": "Mean |loading|"})
    ax.set_title("Cross-table Interaction — Mean |Feature Loading| per CP Component")
    savefig(out_dir, "cross_table_interaction.png")


def plot_variance_decomposition(feature_matrix, feature_names, out_dir):
    bio_idx   = [i for i, n in enumerate(feature_names) if n.startswith("bio_")]
    demo_idx  = [i for i, n in enumerate(feature_names) if n.startswith("demo_")]
    enrol_idx = [i for i, n in enumerate(feature_names) if n.startswith("enrol_")]

    rank = feature_matrix.shape[1]
    # classify each component: which table(s) have >0.2 mean absolute loading?
    threshold = 0.2
    labels_count = {"Bio only": 0, "Demo only": 0, "Enrolment only": 0,
                    "Two tables": 0, "All three": 0}
    for c in range(rank):
        col = np.abs(feature_matrix[:, c])
        active = []
        if bio_idx   and col[bio_idx].mean()   > threshold: active.append("bio")
        if demo_idx  and col[demo_idx].mean()  > threshold: active.append("demo")
        if enrol_idx and col[enrol_idx].mean() > threshold: active.append("enrol")
        if   len(active) == 3: labels_count["All three"]      += 1
        elif len(active) == 2: labels_count["Two tables"]      += 1
        elif "bio"   in active: labels_count["Bio only"]       += 1
        elif "demo"  in active: labels_count["Demo only"]      += 1
        elif "enrol" in active: labels_count["Enrolment only"] += 1
        else:                   labels_count["Two tables"]     += 1  # ambiguous

    labels = [k for k, v in labels_count.items() if v > 0]
    sizes  = [labels_count[k] for k in labels]
    colors_pie = [C["purple"], C["green"], C["orange"], C["amber"], C["pink"]]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=colors_pie[:len(labels)],
        startangle=140, pctdistance=0.75,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("CP Component Variance — Shared vs Table-Specific")
    savefig(out_dir, "variance_decomposition.png")


# ── 3l. Temporal ACF ──────────────────────────────────────────────────────

def plot_temporal_acf(temporal_factors, out_dir, max_lag=20):
    from matplotlib.gridspec import GridSpec
    n_show = min(temporal_factors.shape[1], 4)
    fig = plt.figure(figsize=(12, n_show * 2.5))
    gs  = GridSpec(n_show, 1, figure=fig, hspace=0.5)

    for i in range(n_show):
        ax   = fig.add_subplot(gs[i])
        x    = temporal_factors[:, i]
        T    = len(x)
        lags = range(1, min(max_lag + 1, T // 2))
        acf  = [np.corrcoef(x[:-lag], x[lag:])[0, 1] for lag in lags]
        conf = 1.96 / np.sqrt(T)
        ax.bar(list(lags), acf,
               color=[C["purple"] if abs(v) > conf else C["gray"] for v in acf],
               width=0.6)
        ax.axhline(conf,  color=C["orange"], ls="--", lw=1)
        ax.axhline(-conf, color=C["orange"], ls="--", lw=1)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(f"C{i+1}", rotation=0, labelpad=24, fontsize=9)
        ax.set_ylim(-1.1, 1.1)

    axes_list = fig.get_axes()
    if axes_list:
        axes_list[-1].set_xlabel("Lag (time steps)")
    fig.suptitle("Temporal Factor Autocorrelation (ACF)", fontsize=12)
    savefig(out_dir, "temporal_factor_acf.png")


# ── 3m. Feature radar ─────────────────────────────────────────────────────

def plot_feature_radar(feature_matrix, feature_names, out_dir, n_comp=5):
    n_show = min(n_comp, feature_matrix.shape[1])
    n_feat = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    colors = [C["purple"], C["green"], C["orange"], C["amber"], C["pink"]]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i in range(n_show):
        vals = np.abs(feature_matrix[:, i]).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[i % len(colors)], lw=2, label=f"C{i+1}")
        ax.fill(angles, vals, color=colors[i % len(colors)], alpha=0.08)

    ax.set_xticks(angles[:-1])
    short_names = [n.split("_", 1)[-1][:12] for n in feature_names]
    ax.set_xticklabels(short_names, fontsize=7)
    ax.set_title("CP Feature Loadings — Radar Chart\n(absolute values)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    savefig(out_dir, "cp_feature_radar.png")


def plot_top3_feature_bar(feature_matrix, feature_names, out_dir):
    n_show = min(3, feature_matrix.shape[1])
    x = np.arange(len(feature_names))
    width = 0.25
    colors = [C["purple"], C["green"], C["orange"]]
    fig, ax = plt.subplots(figsize=(max(12, len(feature_names) * 0.55), 5))
    for i in range(n_show):
        ax.bar(x + i * width, feature_matrix[:, i], width,
               color=colors[i], alpha=0.8, label=f"Component {i+1}")
    short_names = [n.split("_", 1)[-1] for n in feature_names]
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Feature Loading")
    ax.set_title("Feature Contributions to Top-3 CP Components")
    ax.legend()
    savefig(out_dir, "cp_feature_top3_bar.png")


# ── 3n. Spatial factor choropleth (simple scatter as fallback) ────────────

def plot_spatial_factor_map(district_factors, districts, meta, component, out_dir):
    """Simple scatter-map using lat/lon if available; else horizontal bar."""
    scores = district_factors[:, component]
    df = pd.DataFrame({"dist_key": districts, "score": scores})
    df = df.merge(meta[["dist_key", "state", "district"]], on="dist_key", how="left")
    df_sorted = df.sort_values("score", ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [C["purple"] if s >= 0 else C["orange"] for s in df_sorted["score"]]
    ax.barh(df_sorted["district"].str[:20], df_sorted["score"],
            color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("CP Factor Score")
    ax.set_title(f"Top 30 Districts — CP Spatial Factor {component+1}")
    ax.invert_yaxis()
    savefig(out_dir, f"cp_spatial_factor{component+1}_map.png")


# ─────────────────────────────────────────────────────────────────────────
# 4.  CSV EXPORTS
# ─────────────────────────────────────────────────────────────────────────

def save_factor_csvs(cp_factors, tucker_core, tucker_factors,
                     districts, dates, feature_names, out_dir):
    rank = cp_factors.factors[0].shape[1]

    # CP district factors
    df_dist = pd.DataFrame(cp_factors.factors[0],
                           columns=[f"cp_c{i+1}" for i in range(rank)])
    df_dist.insert(0, "dist_key", districts)
    df_dist.to_csv(out_dir / "cp_district_factors.csv", index=False)

    # CP temporal factors
    df_time = pd.DataFrame(cp_factors.factors[1],
                           columns=[f"cp_c{i+1}" for i in range(rank)])
    df_time.insert(0, "date", [str(d)[:10] for d in dates])
    df_time.to_csv(out_dir / "cp_temporal_factors.csv", index=False)

    # CP feature loadings
    df_feat = pd.DataFrame(cp_factors.factors[2],
                           columns=[f"cp_c{i+1}" for i in range(rank)])
    df_feat.insert(0, "feature", feature_names)
    df_feat.to_csv(out_dir / "cp_feature_loadings.csv", index=False)

    # Tucker district factors
    r_d = tucker_factors[0].shape[1]
    df_td = pd.DataFrame(tucker_factors[0],
                         columns=[f"tucker_d{i+1}" for i in range(r_d)])
    df_td.insert(0, "dist_key", districts)
    df_td.to_csv(out_dir / "tucker_district_factors.csv", index=False)

    # Tucker temporal factors
    r_t = tucker_factors[1].shape[1]
    df_tt = pd.DataFrame(tucker_factors[1],
                         columns=[f"tucker_t{i+1}" for i in range(r_t)])
    df_tt.insert(0, "date", [str(d)[:10] for d in dates])
    df_tt.to_csv(out_dir / "tucker_temporal_factors.csv", index=False)

    # Tucker feature factors
    r_f = tucker_factors[2].shape[1]
    df_tf = pd.DataFrame(tucker_factors[2],
                         columns=[f"tucker_f{i+1}" for i in range(r_f)])
    df_tf.insert(0, "feature", feature_names)
    df_tf.to_csv(out_dir / "tucker_feature_factors.csv", index=False)

    print("  factor CSVs saved.")


def save_tensor_summary(N, T, F, cp_rank, cp_error,
                        tucker_ranks, tucker_error, out_dir):
    df = pd.DataFrame([{
        "n_districts":    N,
        "n_timesteps":    T,
        "n_features":     F,
        "rank":           cp_rank,
        "cp_error":       round(cp_error,   6),
        "tucker_ranks":   str(tucker_ranks),
        "tucker_error":   round(tucker_error, 6),
    }])
    df.to_csv(out_dir / "tensor_summary.csv", index=False)
    print("  tensor_summary.csv saved.")


# ─────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",           default="../database/aadhar.duckdb",
                        help="Path to the DuckDB database file")
    parser.add_argument("--output",       default="tensor_output",
                        help="Output directory")
    parser.add_argument("--cp-rank",      type=int, default=10,
                        help="CP / PARAFAC rank")
    parser.add_argument("--max-rank",     type=int, default=20,
                        help="Maximum rank to sweep for elbow curve")
    parser.add_argument("--tucker-ranks", type=int, nargs=3, default=None,
                        metavar=("R_DIST", "R_TIME", "R_FEAT"),
                        help="Tucker ranks for [district, time, feature] modes")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & build tensor ────────────────────────────────────────────
    print("\n[1/5] Loading tables from DuckDB …")
    dfs = load_tables(args.db)
    print(f"      bio:   {len(dfs['bio'])} rows")
    print(f"      demo:  {len(dfs['demo'])} rows")
    print(f"      enrol: {len(dfs['enrol'])} rows")
    

    print("\n[2/5] Building tensor …")
    tensor_data, districts, dates, feature_names, meta = build_tensor(dfs)
    N, T, F = tensor_data.shape
    print(f"      Tensor shape: {N} districts × {T} time steps × {F} features")
    print(f"      Features: {feature_names}")

    # ── 2. CP rank sweep ──────────────────────────────────────────────────
    print(f"\n[3/5] Sweeping CP ranks 2 → {args.max_rank} …")
    ranks, errors = sweep_cp_rank(tensor_data, args.max_rank, args.seed)
    plot_cp_error_curve(ranks, errors, args.cp_rank, out_dir)

    # ── 3. Fit CP at chosen rank ──────────────────────────────────────────
    print(f"\n      Fitting CP at rank = {args.cp_rank} …")
    cp_factors, cp_recon, cp_error = run_cp(tensor_data, args.cp_rank, args.seed)
    print(f"      CP reconstruction error: {cp_error:.6f}")

    # ── 4. Tucker ─────────────────────────────────────────────────────────
    if args.tucker_ranks is None:
        tr = [max(2, int(N * 0.6)), max(2, int(T * 0.6)), max(2, int(F * 0.6))]
        # cap to reasonable sizes
        tr = [min(tr[0], 15), min(tr[1], 12), min(tr[2], 12)]
    else:
        tr = args.tucker_ranks
    print(f"\n      Fitting Tucker at ranks = {tr} …")

    # Tucker rank sweep on feature mode
    tucker_errors_dict = {}
    for fr in range(2, min(F + 1, 13)):
        t_ranks = [tr[0], tr[1], fr]
        _, _, _, terr = run_tucker(tensor_data, t_ranks, args.seed)
        tucker_errors_dict[fr] = terr
        print(f"      Tucker feature_rank={fr}  error={terr:.6f}")
    plot_tucker_error(tucker_errors_dict, out_dir)

    tucker_core, tucker_facs, tucker_recon, tucker_error = run_tucker(
        tensor_data, tr, args.seed
    )
    print(f"      Tucker reconstruction error: {tucker_error:.6f}")

    # ── 5. Generate all plots ─────────────────────────────────────────────
    print("\n[4/5] Generating plots …")

    cp_dist_factors = cp_factors.factors[0]  # (N, rank)
    cp_time_factors = cp_factors.factors[1]  # (T, rank)
    cp_feat_factors = cp_factors.factors[2]  # (F, rank)

    tuck_dist_factors = tucker_facs[0]        # (N, R_d)
    tuck_time_factors = tucker_facs[1]        # (T, R_t)
    tuck_feat_factors = tucker_facs[2]        # (F, R_f)

    plot_cp_explained_variance(cp_factors, tensor_data, out_dir)
    plot_cp_factor_weights(cp_factors, out_dir)
    plot_feature_loading_heatmap(cp_feat_factors, feature_names,
                                 "CP Feature Loadings",
                                 "cp_feature_loadings.png", out_dir)
    plot_feature_loading_heatmap(tuck_feat_factors, feature_names,
                                 "Tucker Feature Factors",
                                 "tucker_feature_factors.png", out_dir)
    plot_top_districts(cp_dist_factors, districts, out_dir)
    plot_district_scatter(cp_dist_factors, districts, meta, out_dir)
    plot_tucker_district_embedding(tuck_dist_factors, districts, meta, out_dir)
    plot_temporal_factors(cp_time_factors, dates, out_dir, label="CP")
    plot_tucker_temporal_heatmap(tuck_time_factors, dates, out_dir)
    plot_tucker_core_slice(tucker_core, out_dir)
    plot_cross_table_corr(cp_dist_factors, feature_names, out_dir)
    plot_cross_table_interaction_heatmap(cp_feat_factors, feature_names, out_dir)
    plot_variance_decomposition(cp_feat_factors, feature_names, out_dir)
    plot_temporal_acf(cp_time_factors, out_dir)
    plot_feature_radar(cp_feat_factors, feature_names, out_dir)
    plot_top3_feature_bar(cp_feat_factors, feature_names, out_dir)
    plot_spatial_factor_map(cp_dist_factors, districts, meta, 0, out_dir)
    plot_spatial_factor_map(cp_dist_factors, districts, meta, 1, out_dir)

    # ── 6. Save CSVs ──────────────────────────────────────────────────────
    print("\n[5/5] Saving CSV factor matrices …")
    save_factor_csvs(cp_factors, tucker_core, tucker_facs,
                     districts, dates, feature_names, out_dir)
    save_tensor_summary(N, T, F, args.cp_rank, cp_error,
                        tr, tucker_error, out_dir)

    print(f"\n✓ All outputs written to → {out_dir.resolve()}/\n")


if __name__ == "__main__":
    main()