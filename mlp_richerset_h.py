import os
import copy
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from econometric_models_h import (
    vol_floor,
    load_and_prepare_data,
    plot_volatility_forecasts,
)

##################################################################
# MLP script for forecasting, with a richer feature input set
##################################################################

# -----------------------------
# 0) Global configuration / defaults
# -----------------------------

# Default asset and data settings
asset_name = "DAX"
data_path = "DAX Historical Data.csv"
split_date = pd.Timestamp("2018-01-01")

# Forecast horizons to estimate by default
horizons_default = [21, 63]


# If True, the model is trained on log-volatility rather than level volatility
# This often improves numerical stability and can make the target distribution
# more symmetric.
use_log_target = True
target_col = "log_HV_30d" if use_log_target else "HV_30d"

# Number of past observations included in each input window
look = 60

# Training hyperparameters
seed = 42
max_epochs = 250
patience = 20
batch_size = 32

# Directory for forecast files, plots, and evaluation summaries
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


def get_horizon_config(h):
    """
    Return horizon-specific hyperparameters.

    The idea is that different forecast horizons may require slightly
    different regularization or learning-rate settings.
    """
    if h == 21:
        return {
            "hidden_dims": (64, 32),
            "dropout": 0.10,
            "lr": 7e-4,
            "weight_decay": 1e-5,
            "loss_name": "mse",
        }
    elif h == 63:
        return {
            "hidden_dims": (64, 32),
            "dropout": 0.15,
            "lr": 5e-4,
            "weight_decay": 1e-5,
            "loss_name": "mse",
        }
    else:
        # Fallback configuration for any other horizon
        return {
            "hidden_dims": (64, 32),
            "dropout": 0.10,
            "lr": 5e-4,
            "weight_decay": 1e-5,
            "loss_name": "mse",
        }


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed=42):
    """
    Set random seeds across Python, NumPy, and PyTorch to improve reproducibility.

    This also enforces deterministic CuDNN behavior when available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_loss(loss_name="mse"):
    """
    Map a loss name to the corresponding PyTorch loss object.
    """
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name in ("smoothl1", "huber"):
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {loss_name}")


def qlike_loss(y_true, y_pred, eps=1e-12):
    """
    Compute the QLIKE loss for volatility forecasts.

    For volatility forecasts, QLIKE is often preferred in addition to
    squared-error metrics because it penalizes volatility miscalibration
    in a different and financially meaningful way.

    Formula:
        QLIKE = mean( log(sigma_hat^2) + sigma_true^2 / sigma_hat^2 )
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Prevent zero or negative values before squaring/logging
    y_true = np.clip(y_true, eps, None)
    y_pred = np.clip(y_pred, eps, None)

    sigma2_true = y_true ** 2
    sigma2_pred = y_pred ** 2

    return np.mean(np.log(sigma2_pred) + sigma2_true / sigma2_pred)


def evaluate_forecasts(y_true, y_pred):
    """
    Compute standard forecast evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    qlike = qlike_loss(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "QLIKE": qlike,
    }


def parse_args():
    """
    Parse command-line arguments so the script can also be run flexibly
    from the terminal with different assets, horizons, or optional features.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-name", type=str, default=asset_name)
    ap.add_argument("--data-path", type=str, default=data_path)
    ap.add_argument("--split-date", type=str, default=str(split_date.date()))
    ap.add_argument("--horizons", type=int, nargs="+", default=horizons_default)

    return ap.parse_args()


# -----------------------------
# 2) Feature engineering
# -----------------------------

def add_features(df):
    """
    Create volatility-related predictors from the raw input data.

    The feature set combines:
    - persistence features of historical volatility,
    - short-horizon return-based realized variance proxies,
    - downside risk measures,
    - EWMA-style volatility information,
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Ensure strictly positive volatility before taking logs
    df["HV_30d"] = df["HV_30d"].clip(lower=vol_floor)
    df["log_HV_30d"] = np.log(df["HV_30d"])

    # Persistence / regime information:
    # current volatility often depends strongly on recent volatility
    df["HV_30d_lag1"] = df["HV_30d"].shift(1)
    df["log_HV_30d_lag1"] = df["log_HV_30d"].shift(1)

    # Short moving average of historical volatility
    df["HV_30d_ma5"] = df["HV_30d"].rolling(window=5, min_periods=1).mean()
    df["log_HV_30d_ma5"] = np.log(df["HV_30d_ma5"].clip(lower=vol_floor))

    # Return-based information:
    # absolute and squared returns are common volatility proxies
    df["abs_LogReturn"] = df["LogReturn"].abs()
    df["sq_LogReturn"] = df["LogReturn"] ** 2

    # Rolling realized-variance-style measures over different windows
    df["rv_5d"] = df["sq_LogReturn"].rolling(window=5, min_periods=1).mean()
    df["rv_10d"] = df["sq_LogReturn"].rolling(window=10, min_periods=1).mean()
    df["rv_21d"] = df["sq_LogReturn"].rolling(window=21, min_periods=1).mean()
    df["rv_63d"] = df["sq_LogReturn"].rolling(window=63, min_periods=1).mean()

    # Log-transform rolling variance measures
    df["log_rv_5d"] = np.log(df["rv_5d"].clip(lower=vol_floor))
    df["log_rv_10d"] = np.log(df["rv_10d"].clip(lower=vol_floor))
    df["log_rv_21d"] = np.log(df["rv_21d"].clip(lower=vol_floor))
    df["log_rv_63d"] = np.log(df["rv_63d"].clip(lower=vol_floor))

    # Downside realized variance:
    # only negative returns contribute, which captures asymmetric risk
    neg_ret = df["LogReturn"].clip(upper=0.0)
    df["downside_rv_21d"] = (neg_ret ** 2).rolling(window=21, min_periods=1).mean()
    df["log_downside_rv_21d"] = np.log(df["downside_rv_21d"].clip(lower=vol_floor))

    # EWMA variance estimate, then converted to log-volatility
    df["ewma_var"] = df["sq_LogReturn"].ewm(alpha=1 - 0.94, adjust=False).mean()
    df["log_ewma_vol"] = 0.5 * np.log(df["ewma_var"].clip(lower=vol_floor))


    return df


def build_model_frame(df, feature_cols, target_col):
    """
    Keep only the columns required for modeling and drop rows with
    missing values introduced by lagging, rolling windows, or merges.
    """
    cols = ["Date"] + feature_cols
    if target_col not in cols:
        cols.append(target_col)
    return df[cols].dropna().reset_index(drop=True)


# -----------------------------
# Flat-window builder for MLP
# -----------------------------

def make_flat_windows(df, feature_cols, target_col, look, h):
    """
    Construct rolling input windows for direct h-step-ahead forecasting.

    For each time index i:
    - use observations from [i-look+1, ..., i] as the input window,
    - predict the target located at i+h.

    Since this is an MLP and not a sequence model, each 2D window
    of shape (look, n_features) is flattened into a 1D feature vector.
    """
    values = df.copy().reset_index(drop=True)

    X, y, target_dates = [], [], []

    for i in range(look - 1, len(values) - h):
        # Window of past feature values up to time i
        x_window = values.iloc[i - look + 1:i + 1][feature_cols].to_numpy(dtype=float)

        # Flatten the full window into one long input vector
        x_flat = x_window.reshape(-1)

        # Direct h-step-ahead target
        y_target = float(values.at[i + h, target_col])

        # Store the calendar date of the forecast target
        d_target = pd.to_datetime(values.at[i + h, "Date"])

        X.append(x_flat)
        y.append(y_target)
        target_dates.append(d_target)

    return np.array(X), np.array(y), pd.to_datetime(target_dates)


# -----------------------------
# Model
# -----------------------------

class MLPForecaster(nn.Module):
    """
    Feedforward neural network for direct volatility forecasting.

    Architecture:
    input -> hidden layers with ReLU + dropout -> scalar output
    """
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers dynamically from the supplied tuple
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hd

        # Final output layer returns one forecast value
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def fit_predict_mlp(
    df_model,
    feature_cols,
    target_col,
    look,
    split_date,
    h,
    seed=42,
    batch_size=32,
    max_epochs=250,
    patience=20,
    lr=5e-4,
    weight_decay=1e-5,
    hidden_dims=(64, 32),
    dropout=0.1,
    loss_name="mse",
    backtransform_from_log=True,
):
    """
    Fit the MLP on the training sample and generate forecasts for the test sample.

    Workflow:
    1. Construct rolling windows.
    2. Split into train/test using the target date.
    3. Split the training set further into train/validation in time order.
    4. Standardize inputs using the core training subset only.
    5. Train with early stopping based on validation loss.
    6. Predict on the test set.
    7. Back-transform forecasts if the model was trained on log-volatility.
    """
    X, y, y_dates = make_flat_windows(
        df=df_model,
        feature_cols=feature_cols,
        target_col=target_col,
        look=look,
        h=h,
    )

    # Train/test split is based on the target date, not the window end date
    train_mask = y_dates < pd.Timestamp(split_date)
    test_mask = y_dates >= pd.Timestamp(split_date)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_dates = y_dates[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Empty train or test set after window construction.")

    # Reserve the last 20% of the training sample as a validation set.
    # This is done in time order, so no future information leaks backwards.
    n_total_train = len(X_train)
    n_train_core = int(np.floor(0.8 * n_total_train))

    if n_train_core <= 0 or n_train_core >= n_total_train:
        raise ValueError("Training split too small for time-ordered validation.")

    X_tr = X_train[:n_train_core]
    y_tr = y_train[:n_train_core]
    X_val = X_train[n_train_core:]
    y_val = y_train[n_train_core:]

    # Standardize predictors using training-core statistics only
    scaler = StandardScaler()
    scaler.fit(X_tr)

    X_tr = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reproducibility and device selection
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert arrays to tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # DataLoader for mini-batch gradient descent
    # shuffle=False preserves time order inside the training sample
    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize model, loss function, and optimizer
    model = MLPForecaster(
        input_dim=X_tr.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    criterion = get_loss(loss_name)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Early-stopping state
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_train_loss = 0.0

        # ---------------------
        # Training pass
        # ---------------------
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            # Gradient clipping can stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_train_loss += loss.item() * xb.size(0)

        train_loss = running_train_loss / max(1, len(X_tr_t))

        # ---------------------
        # Validation pass
        # ---------------------
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device))
            val_loss = criterion(val_pred, y_val_t.to(device)).item()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        # Save the best model according to validation loss
        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.6f})")
                break

    # Restore best validation model before forecasting
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------------
    # Test prediction
    # ---------------------
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t.to(device)).detach().cpu().numpy().reshape(-1)

    # Convert predictions back to level volatility if the target was logged
    if backtransform_from_log:
        y_pred_level = np.exp(y_pred)
        y_test_level = np.exp(y_test)
    else:
        y_pred_level = y_pred.copy()
        y_test_level = y_test.copy()

    # Enforce a lower bound on predicted volatility
    y_pred_level = np.clip(y_pred_level, vol_floor, None)

    # Return test-period realized volatility and forecasts in one DataFrame
    out = pd.DataFrame({
        "Date": test_dates,
        "HV_30d": y_test_level,
        "Forecast": y_pred_level,
    })

    return out


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    """
    End-to-end workflow:
    - read command-line arguments,
    - load and preprocess the data,
    - create features,
    - train one MLP per forecast horizon,
    - evaluate forecasts,
    - save output files and plots.
    """
    global asset_name, data_path, split_date

    args = parse_args()

    asset_name = args.asset_name
    data_path = args.data_path
    split_date = pd.Timestamp(args.split_date)
    horizons = args.horizons

    # -------------------------
    # Load data
    # -------------------------
    df = load_and_prepare_data(
        path=data_path,
        sep=";",
        decimal=",",
        date_format="%m.%d.%Y",
        drop_cols=["Open", "High", "Low", "Vol,", "Change %"],
    )


    # Ensure proper date format and chronological sorting
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # -------------------------
    # Feature engineering
    # -------------------------
    df = add_features(df)

    # Baseline feature set used by the MLP
    plain_feature_cols = [
        "log_HV_30d",
        "abs_LogReturn",
        "sq_LogReturn",
        "log_HV_30d_lag1",
        "log_HV_30d_ma5",
        "log_rv_5d",
        "log_rv_10d",
        "log_rv_21d",
        "log_rv_63d",
        "log_downside_rv_21d",
        "log_ewma_vol",
    ]


    # Final modeling DataFrame
    df_model_plain = build_model_frame(df, plain_feature_cols, target_col)

    # Diagnostic check for duplicate column names
    print("plain duplicates:", df_model_plain.columns[df_model_plain.columns.duplicated()].tolist())

    eval_rows = []

    # -------------------------
    # Horizon loop
    # -------------------------
    for h in horizons:
        cfg = get_horizon_config(h)

        print("\n" + "=" * 70)
        print(f"Training MLP for horizon h={h}")
        print("=" * 70)

        # Fit the model and generate out-of-sample forecasts for horizon h
        out = fit_predict_mlp(
            df_model=df_model_plain,
            feature_cols=plain_feature_cols,
            target_col=target_col,
            look=look,
            split_date=split_date,
            h=h,
            seed=seed,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
            loss_name=cfg["loss_name"],
            backtransform_from_log=use_log_target,
        ).rename(columns={"Forecast": f"MLP_Forecast_h{h}"})

        # Evaluate forecast accuracy for this horizon
        metrics = evaluate_forecasts(
            y_true=out["HV_30d"].values,
            y_pred=out[f"MLP_Forecast_h{h}"].values,
        )
        metrics["Model"] = f"MLP_Forecast_h{h}"
        metrics["Horizon"] = h
        eval_rows.append(metrics)

        # Save the forecast file for this horizon
        forecast_path = os.path.join(output_dir, f"mlp_forecasts_h{h}.csv")
        out.to_csv(forecast_path, index=False, sep=";")
        print(f"Wrote forecast file: {forecast_path}")

        # Plot realized volatility and model forecast
        plot_path = os.path.join(output_dir, f"mlp_plot_h{h}.png")
        plot_volatility_forecasts(
            df=out,
            forecast_specs=[(f"MLP_Forecast_h{h}", "MLP")],
            title=f"{asset_name}: HV_30d vs MLP forecast (k={h})",
            split_date=split_date,
            save_path=plot_path,
            show=True,
        )
        print(f"Wrote plot file: {plot_path}")

    # -------------------------
    # Save evaluation summary
    # -------------------------
    eval_df = pd.DataFrame(eval_rows)[["Horizon", "Model", "MSE", "RMSE", "MAE", "QLIKE"]]
    eval_df = eval_df.sort_values(["Horizon", "QLIKE"]).reset_index(drop=True)

    print("\nEvaluation summary:")
    print(eval_df)

    eval_path = os.path.join(output_dir, "mlp_evaluation_summary.csv")
    eval_df.to_csv(eval_path, index=False, sep=";")
    print(f"Wrote evaluation file: {eval_path}")


if __name__ == "__main__":
    main()


# =========================================================
# Example usage
# =========================================================
#
# Default run:
#   python mlp_h.py
#
# Specific horizons:
#   python mlp_h.py --horizons 21 63
#
# Another asset:
#   python mlp_h.py --asset-name "BMW" --data-path "BMW Historical Data.csv" --horizons 21 63