import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac

tl.set_backend("numpy")


# =========================================================
# HELPERS
# =========================================================

def load_metadata(tensor_path, week_index_path, district_path, feature_path):
    X = np.load(tensor_path).astype(np.float32)

    week_df = pd.read_csv(week_index_path)
    week_col = week_df.columns[0]
    weeks = pd.to_datetime(week_df[week_col]).astype(str).tolist()

    district_df = pd.read_csv(district_path)
    districts = (
        district_df["district"].tolist()
        if "district" in district_df.columns
        else district_df.iloc[:, 1].tolist()
    )

    feature_df = pd.read_csv(feature_path)
    features = feature_df.iloc[:, 0].tolist()

    return X, weeks, districts, features


def standardize_tensor(X):
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    return Xs, mean, std


def relative_reconstruction_error(X, X_hat):
    num = np.linalg.norm(X - X_hat)
    den = np.linalg.norm(X) + 1e-8
    return num / den


def save_heatmap(df, title, output_path, cmap="viridis", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#f9f8f6")
    ax.set_facecolor("#ffffff")
    im = ax.imshow(df.values, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#f9f8f6")
    plt.close(fig)


def save_time_plot(df, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#f9f8f6")
    ax.set_facecolor("#ffffff")

    for col in df.columns:
        ax.plot(pd.to_datetime(df.index), df[col], linewidth=2, label=col)

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("Week")
    ax.set_ylabel("Factor loading")
    ax.grid(True, color="#f0f0f0")
    ax.legend(ncol=2, fontsize=9)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#e5e7eb")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#f9f8f6")
    plt.close(fig)


# =========================================================
# CORE
# =========================================================

def run_cp_decomposition(X, rank, random_state=42, n_iter_max=300):
    cp_tensor = parafac(
        X,
        rank=rank,
        init="svd",
        random_state=random_state,
        n_iter_max=n_iter_max,
        normalize_factors=False,
        tol=1e-7,
    )
    weights, factors = cp_tensor
    X_hat = tl.cp_to_tensor(cp_tensor)
    return weights, factors, X_hat


def process_tensor(
    name,
    tensor_path,
    week_index_path,
    district_path,
    feature_path,
    output_dir,
    rank=3,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    X, weeks, districts, features = load_metadata(
        tensor_path, week_index_path, district_path, feature_path
    )

    Xs, mean, std = standardize_tensor(X)
    weights, factors, X_hat = run_cp_decomposition(Xs, rank=rank)

    time_f, district_f, feature_f = factors

    comp_cols = [f"Component_{i+1}" for i in range(rank)]

    time_df = pd.DataFrame(time_f, index=weeks, columns=comp_cols)
    district_df = pd.DataFrame(district_f, index=districts, columns=comp_cols)
    feature_df = pd.DataFrame(feature_f, index=features, columns=comp_cols)
    weight_df = pd.DataFrame({
        "component": comp_cols,
        "weight": weights
    }).sort_values("weight", ascending=False)

    rr_error = relative_reconstruction_error(Xs, X_hat)
    fit = 1.0 - rr_error

    time_df.to_csv(output_dir / "time_factors.csv")
    district_df.to_csv(output_dir / "district_factors.csv")
    feature_df.to_csv(output_dir / "feature_factors.csv")
    weight_df.to_csv(output_dir / "component_strengths.csv", index=False)

    save_time_plot(
        time_df,
        f"{name.title()} tensor decomposition — time factors",
        output_dir / "time_factors.png",
    )

    save_heatmap(
        district_df,
        f"{name.title()} tensor decomposition — district factors",
        output_dir / "district_component_heatmap.png",
        cmap="magma",
        figsize=(11, 8),
    )

    save_heatmap(
        feature_df,
        f"{name.title()} tensor decomposition — feature factors",
        output_dir / "feature_component_heatmap.png",
        cmap="viridis",
        figsize=(8, 4.5),
    )

    # top districts per component
    top_rows = []
    for comp in comp_cols:
        top_d = district_df[comp].abs().sort_values(ascending=False).head(10)
        for district, value in top_d.items():
            top_rows.append({
                "component": comp,
                "district": district,
                "loading": float(value),
            })
    pd.DataFrame(top_rows).to_csv(output_dir / "top_districts_by_component.csv", index=False)

    # top features per component
    feat_rows = []
    for comp in comp_cols:
        top_f = feature_df[comp].abs().sort_values(ascending=False)
        for feat, value in top_f.items():
            feat_rows.append({
                "component": comp,
                "feature": feat,
                "loading": float(value),
            })
    pd.DataFrame(feat_rows).to_csv(output_dir / "feature_rankings_by_component.csv", index=False)

    metrics_text = (
        f"Tensor name = {name}\n"
        f"Shape = {X.shape}\n"
        f"Rank = {rank}\n"
        f"Relative reconstruction error = {rr_error:.6f}\n"
        f"Approximate fit = {fit:.6f}\n"
    )
    (output_dir / "reconstruction_metrics.txt").write_text(metrics_text, encoding="utf-8")


def main(args):
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    district_path = Path(args.districts)

    process_tensor(
        name="biometric",
        tensor_path=Path(args.bio_tensor),
        week_index_path=Path(args.bio_week_index),
        district_path=district_path,
        feature_path=Path(args.bio_features),
        output_dir=base_out / "biometric",
        rank=args.rank,
    )

    process_tensor(
        name="enrolment",
        tensor_path=Path(args.enrol_tensor),
        week_index_path=Path(args.enrol_week_index),
        district_path=district_path,
        feature_path=Path(args.enrol_features),
        output_dir=base_out / "enrolment",
        rank=args.rank,
    )

    print(f"Saved tensor decomposition outputs to: {base_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bio_tensor", default="../NEW_STGCN/biometric_weekly_output/feature_tensor_X.npy")
    p.add_argument("--bio_week_index", default="../NEW_STGCN/biometric_weekly_output/week_index.csv")
    p.add_argument("--bio_features", default="../NEW_STGCN/biometric_weekly_output/tensor_feature_columns.csv")

    p.add_argument("--enrol_tensor", default="../NEW_STGCN/enrolment_weekly_output/feature_tensor_X.npy")
    p.add_argument("--enrol_week_index", default="../NEW_STGCN/enrolment_weekly_output/week_index.csv")
    p.add_argument("--enrol_features", default="../NEW_STGCN/enrolment_weekly_output/tensor_feature_columns.csv")

    p.add_argument("--districts", default="../NEW_STGCN/graph_output/district_order.csv")
    p.add_argument("--output_dir", default="tensor_output")
    p.add_argument("--rank", type=int, default=3)

    args = p.parse_args()
    main(args)