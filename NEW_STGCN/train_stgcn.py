import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# IO
# =========================================================

def load_data(tensor_path, laplacian_path, district_path, week_index_path):
    X = np.load(tensor_path).astype(np.float32)  # [T, N, C]
    L = pd.read_csv(laplacian_path, index_col=0).values.astype(np.float32)

    districts_df = pd.read_csv(district_path)
    districts = (
        districts_df["district"].tolist()
        if "district" in districts_df.columns
        else districts_df.iloc[:, 1].tolist()
    )

    week_df = pd.read_csv(week_index_path)
    week_col = week_df.columns[0]
    weeks = pd.to_datetime(week_df[week_col]).tolist()

    lambda_max = np.linalg.eigvalsh(L).max()
    L_tilde = (2.0 / (lambda_max + 1e-8)) * L - np.eye(L.shape[0], dtype=np.float32)

    return X, L_tilde, districts, weeks


# =========================================================
# SEQUENCES
# =========================================================

def make_sequences(X, weeks, target_idx, t_in=6, t_out=1):
    """
    Inputs:
      X      : [T, N, C]
      weeks  : length T
    Returns:
      inputs       : [S, t_in, N, C]
      targets      : [S, t_out, N, 1]
      target_weeks : [S, t_out]  (timestamps for prediction horizon)
    """
    inputs, targets, target_weeks = [], [], []
    T = X.shape[0]

    for t in range(T - t_in - t_out + 1):
        inputs.append(X[t:t+t_in])  # [t_in, N, C]
        y = X[t+t_in:t+t_in+t_out, :, target_idx:target_idx+1]
        inputs_weeks = weeks[t:t+t_in]
        future_weeks = weeks[t+t_in:t+t_in+t_out]

        targets.append(y)
        target_weeks.append(future_weeks)

    return np.array(inputs), np.array(targets), np.array(target_weeks, dtype=object)


def chronological_split(inputs, targets, target_weeks, train_r=0.6, val_r=0.2):
    S = len(inputs)
    s1 = int(S * train_r)
    s2 = int(S * (train_r + val_r))

    return (
        inputs[:s1], targets[:s1], target_weeks[:s1],
        inputs[s1:s2], targets[s1:s2], target_weeks[s1:s2],
        inputs[s2:], targets[s2:], target_weeks[s2:]
    )


# =========================================================
# SCALERS
# =========================================================

class FeatureScaler:
    def fit(self, X):
        # X: [S, T, N, C]
        self.mean = X.mean(axis=(0, 1, 2), keepdims=True)
        self.std = X.std(axis=(0, 1, 2), keepdims=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean


class TargetScaler:
    def fit(self, y):
        # y: [S, T_out, N, 1]
        self.mean = y.mean(axis=(0, 1, 2), keepdims=True)
        self.std = y.std(axis=(0, 1, 2), keepdims=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        return y * self.std + self.mean


# =========================================================
# MODEL
# =========================================================

class ChebConv(nn.Module):
    def __init__(self, c_in, c_out, K, L_tilde):
        super().__init__()
        self.K = K
        self.c_out = c_out
        self.register_buffer("L_tilde", torch.tensor(L_tilde, dtype=torch.float32))
        self.weight = nn.Parameter(torch.FloatTensor(K, c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [B, T, N, C]
        B, T, N, C = x.shape
        x_flat = x.reshape(B * T, N, C)

        Tx_0 = x_flat
        out = torch.einsum("bnc,co->bno", Tx_0, self.weight[0])

        if self.K > 1:
            Tx_1 = torch.bmm(
                self.L_tilde.unsqueeze(0).expand(B * T, -1, -1),
                x_flat
            )
            out = out + torch.einsum("bnc,co->bno", Tx_1, self.weight[1])

            for k in range(2, self.K):
                Tx_2 = 2 * torch.bmm(
                    self.L_tilde.unsqueeze(0).expand(B * T, -1, -1),
                    Tx_1
                ) - Tx_0
                out = out + torch.einsum("bnc,co->bno", Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2

        out = out + self.bias
        return out.reshape(B, T, N, self.c_out)


class TemporalConv(nn.Module):
    def __init__(self, c_in, c_out, Kt):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(Kt, 1))

    def forward(self, x):
        # [B, T, N, C] -> [B, C, T, N]
        x = x.permute(0, 3, 1, 2)
        out = self.conv(x)
        P, Q = out.chunk(2, dim=1)
        out = P * torch.sigmoid(Q)
        return out.permute(0, 2, 3, 1)  # [B, T', N, C]


class STConvBlock(nn.Module):
    def __init__(self, c_in, c_spatial, c_out, Kt, K, L_tilde):
        super().__init__()
        self.t1 = TemporalConv(c_in, c_out, Kt)
        self.gc = ChebConv(c_out, c_spatial, K, L_tilde)
        self.t2 = TemporalConv(c_spatial, c_out, Kt)
        self.res = nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()
        self.norm = nn.LayerNorm(c_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        T_in = x.shape[1]
        res = x

        out = self.t1(x)
        out = self.relu(self.gc(out))
        out = self.t2(out)

        T_out = out.shape[1]
        trim = T_in - T_out
        res = res[:, trim // 2: T_in - (trim - trim // 2), :, :]
        res = self.res(res)

        return self.norm(out + res)


class STGCN(nn.Module):
    def __init__(self, C, T_out, K, Kt, L_tilde, c_s=16, c_out=64):
        super().__init__()
        self.t_out_steps = T_out
        self.block1 = STConvBlock(C, c_s, c_out, Kt, K, L_tilde)
        self.block2 = STConvBlock(c_out, c_s, c_out, Kt, K, L_tilde)
        self.t_out = TemporalConv(c_out, c_out, Kt)
        self.fc = nn.Linear(c_out, 1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.t_out(out)
        out = self.fc(out)
        return out[:, -self.t_out_steps:, :, :]


# =========================================================
# TRAIN / EVAL
# =========================================================

def train_model(model, tr_x, tr_y, va_x, va_y, epochs=100, lr=1e-3):
    tr_x = torch.tensor(tr_x, dtype=torch.float32, device=DEVICE)
    tr_y = torch.tensor(tr_y, dtype=torch.float32, device=DEVICE)
    va_x = torch.tensor(va_x, dtype=torch.float32, device=DEVICE)
    va_y = torch.tensor(va_y, dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None
    patience = 20
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(tr_x)
        loss = F.mse_loss(pred, tr_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vpred = model(va_x)
            vloss = F.mse_loss(vpred, va_y).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train={loss.item():.5f} | val={vloss:.5f}")

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return model


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mae, rmse, mape


def save_prediction_csv(pred_denorm, true_denorm, test_weeks, districts, output_csv):
    """
    pred_denorm, true_denorm: [S_test, T_out, N, 1]
    test_weeks: [S_test, T_out]
    """
    rows = []
    S, T_out, N, _ = pred_denorm.shape

    for s in range(S):
        for h in range(T_out):
            week_val = pd.to_datetime(test_weeks[s, h])
            for n in range(N):
                rows.append({
                    "week_start": week_val,
                    "district": districts[n],
                    "actual": float(true_denorm[s, h, n, 0]),
                    "predicted": float(pred_denorm[s, h, n, 0]),
                    "abs_error": float(abs(true_denorm[s, h, n, 0] - pred_denorm[s, h, n, 0])),
                })

    pred_df = pd.DataFrame(rows).sort_values(["week_start", "district"])
    pred_df.to_csv(output_csv, index=False)
    return pred_df


def evaluate(model, te_x, te_y_scaled, te_y_raw, test_weeks, districts, target_scaler, output_dir):
    model.eval()
    te_x_t = torch.tensor(te_x, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred_scaled = model(te_x_t).cpu().numpy()

    pred_denorm = target_scaler.inverse_transform(pred_scaled)
    true_denorm = te_y_raw

    # flattened overall metrics
    p = pred_denorm[:, :, :, 0].reshape(-1)
    y = true_denorm[:, :, :, 0].reshape(-1)
    mae, rmse, mape = compute_metrics(y, p)

    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAPE = {mape:.2f}%")

    # save arrays
    np.save(output_dir / "predictions_denorm.npy", pred_denorm)
    np.save(output_dir / "actuals_denorm.npy", true_denorm)
    np.save(output_dir / "predictions_scaled.npy", pred_scaled)
    np.save(output_dir / "actuals_scaled.npy", te_y_scaled)

    # save long csv
    pred_df = save_prediction_csv(
        pred_denorm=pred_denorm,
        true_denorm=true_denorm,
        test_weeks=test_weeks,
        districts=districts,
        output_csv=output_dir / "predictions_by_district_week.csv",
    )

    # per-district metrics
    district_rows = []
    N = len(districts)
    for n in range(N):
        y_n = true_denorm[:, :, n, 0].reshape(-1)
        p_n = pred_denorm[:, :, n, 0].reshape(-1)
        d_mae, d_rmse, d_mape = compute_metrics(y_n, p_n)
        district_rows.append({
            "district": districts[n],
            "mae": d_mae,
            "rmse": d_rmse,
            "mape": d_mape,
        })

    district_metrics = pd.DataFrame(district_rows).sort_values("mae", ascending=False)
    district_metrics.to_csv(output_dir / "district_metrics.csv", index=False)

    metrics_text = (
        f"MAE  = {mae:.6f}\n"
        f"RMSE = {rmse:.6f}\n"
        f"MAPE = {mape:.4f}%\n"
    )
    (output_dir / "metrics.txt").write_text(metrics_text)

    return pred_df, district_metrics


# =========================================================
# MAIN
# =========================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, L_tilde, districts, weeks = load_data(
        args.tensor, args.laplacian, args.districts, args.week_index
    )

    feature_df = pd.read_csv(args.features)
    feature_names = feature_df.iloc[:, 0].tolist()
    target_idx = feature_names.index(args.target)

    inputs, targets_raw, target_weeks = make_sequences(
        X, weeks, target_idx, t_in=args.t_in, t_out=args.t_out
    )

    (
        tr_x, tr_y_raw, tr_weeks,
        va_x, va_y_raw, va_weeks,
        te_x, te_y_raw, te_weeks
    ) = chronological_split(inputs, targets_raw, target_weeks)

    # fit scalers only on train
    x_scaler = FeatureScaler().fit(tr_x)
    y_scaler = TargetScaler().fit(tr_y_raw)

    tr_x = x_scaler.transform(tr_x)
    va_x = x_scaler.transform(va_x)
    te_x = x_scaler.transform(te_x)

    tr_y = y_scaler.transform(tr_y_raw)
    va_y = y_scaler.transform(va_y_raw)
    te_y_scaled = y_scaler.transform(te_y_raw)

    model = STGCN(
        C=X.shape[2],
        T_out=args.t_out,
        K=args.k,
        Kt=args.kt,
        L_tilde=L_tilde,
        c_s=args.c_spatial,
        c_out=args.c_out,
    ).to(DEVICE)

    model = train_model(
        model, tr_x, tr_y, va_x, va_y,
        epochs=args.epochs, lr=args.lr
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "target": args.target,
            "feature_names": feature_names,
            "districts": districts,
            "t_in": args.t_in,
            "t_out": args.t_out,
            "k": args.k,
            "kt": args.kt,
        },
        output_dir / "best_model.pt"
    )

    pred_df, district_metrics = evaluate(
        model=model,
        te_x=te_x,
        te_y_scaled=te_y_scaled,
        te_y_raw=te_y_raw,
        test_weeks=te_weeks,
        districts=districts,
        target_scaler=y_scaler,
        output_dir=output_dir,
    )

    print(f"Saved -> {output_dir / 'predictions_by_district_week.csv'}")
    print(f"Saved -> {output_dir / 'district_metrics.csv'}")
    print(f"Saved -> {output_dir / 'metrics.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tensor", required=True)
    p.add_argument("--laplacian", required=True)
    p.add_argument("--districts", required=True)
    p.add_argument("--week_index", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--t_in", type=int, default=6)
    p.add_argument("--t_out", type=int, default=1)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--kt", type=int, default=2)
    p.add_argument("--c_spatial", type=int, default=16)
    p.add_argument("--c_out", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)

    args = p.parse_args()
    main(args)