import os
import copy
import random
import numpy as np
import pandas as pd
from arch import arch_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler

from econometric_models_h import (
    vol_floor,
    load_and_prepare_data,
)
from plot_forecasts import plot_volatility_forecasts

##################################################################
# script for forecasting volatility with pure lstm and hybrid mdoels
# for the k-step-ahead step
##################################################################


# -----------------------------
# configuration / defaults
# -----------------------------

asset_name = "USD/EUR"
data_path = "USD_EUR Historical Data.csv"
split_date = pd.Timestamp("2018-01-01")
horizon = 63
use_log_target = True
target_col = "log_HV_30d" if use_log_target else "HV_30d"

# sequence / model
look = 60
seed = 42
max_epochs = 250
patience = 20
batch_size = 32

# horizon-specific defaults
def get_horizon_config(horizon):
    """
    Retrun a set of default hyperparameters depending on the forecast horizon.
    """
    if horizon == 21:
        return {
            "hidden_size": 32,
            "fc_size": 16,
            "dropout": 0.05,
            "lr": 7e-4,
            "weight_decay": 1e-5,
            "loss_name": "mse",
        }
    elif horizon == 63:
        return {
            "hidden_size": 24,
            "fc_size": 12,
            "dropout": 0.10,
            "lr": 5e-4,
            "weight_decay": 1e-5,
            "loss_name": "mse",
        }
    else:
        return {
            "hidden_size": 24,
            "fc_size": 12,
            "dropout": 0.10,
            "lr": 5e-4,
            "weight_decay": 1e-5,
            "loss_name": "mse",
        }

# output dictonary for forecast and evaluation files
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed=42):
    """
    Set random seetds for NumPy and PyTorch to improve reproducibilty
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_loss(loss_name="mse"):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name in ("smoothl1", "huber"):
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {loss_name}")


"""
def qlike_loss(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true = np.clip(y_true, eps, None)
    y_pred = np.clip(y_pred, eps, None)

    sigma2_true = y_true ** 2
    sigma2_pred = y_pred ** 2

    return np.mean(np.log(sigma2_pred) + sigma2_true / sigma2_pred)


def evaluate_forecasts(y_true, y_pred):
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
    }"""


def parse_args():
    """
    Parse optional command-line arguments so the script can be reused for different
    assets, horizons.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=horizon)
    ap.add_argument("--asset-name", type=str, default=asset_name)
    ap.add_argument("--data-path", type=str, default=data_path)
    ap.add_argument("--split-date", type=str, default=str(split_date.date()))
    return ap.parse_args()

# -----------------------------
# Econometric origin-dated forecastss
# -----------------------------

def add_garch_origin_forecast(
    df,
    split_date="2018-01-01",
    h=21,
    scale=100.0,
    forecast_col=None,
    mean="Constant",
    dist="normal",
):
    """
    Creates h-step-ahead GARCH forecast made at time t and stores it at row t.
    Parameters are estimated only on the training sample, then applied
    recursively over the full dataset.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    split_ts = pd.Timestamp(split_date)

    if forecast_col is None:
        forecast_col = f"GARCH_Origin_Forecast_h{h}"

    r = (df["LogReturn"] * scale).astype(float)
    train_mask = df["Date"] < split_ts
    returns_train = r.loc[train_mask].dropna()

    df[forecast_col] = np.nan

    if len(returns_train) < 50:
        return df

    model = arch_model(
        returns_train,
        vol="GARCH",
        p=1,
        q=1,
        mean=mean,
        dist=dist,
    )
    res = model.fit(disp="off")

    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    phi = alpha + beta
    mu = float(params["mu"]) if "mu" in params.index else 0.0

    # unconditional variance if process stationary
    longrun = np.nan
    if phi < 1.0:
        longrun = omega / (1.0 - phi)

    n = len(df)

    # container for the filtered conditional variance over the full sample
    sigma2 = np.full(n, np.nan)

    # fill in fitted in-sample conditional variances from the estimated model
    cond_vol_train = res.conditional_volatility
    for pos in returns_train.index.to_numpy():
        sigma2[int(pos)] = float(cond_vol_train.loc[pos] ** 2)

    # continue recursion through the full sample using fixed train parameters
    last_train_pos = int(returns_train.index.max())

    for t in range(last_train_pos + 1, n):
        prev_sigma2 = sigma2[t - 1]
        if np.isnan(prev_sigma2):
            if np.isnan(longrun):
                prev_sigma2 = np.var(returns_train)
            else:
                prev_sigma2 = longrun

        prev_eps = float(r.iloc[t - 1] - mu)
        sigma2[t] = omega + alpha * (prev_eps ** 2) + beta * prev_sigma2

    # create h-step-ahead forecast at each forecast origin t
    # only where target t+h exists
    for t in range(n - h):
        sigma2_t = sigma2[t]
        if np.isnan(sigma2_t):
            continue

        eps_t = float(r.iloc[t] - mu)
        sigma2_1_ahead = omega + alpha * (eps_t ** 2) + beta * sigma2_t

        # for h >1, use mean reversion toward the unconditional variance
        if np.isnan(longrun):
            sigma2_h = sigma2_1_ahead
        else:
            sigma2_h = longrun + (phi ** (h - 1)) * (sigma2_1_ahead - longrun)

        df.at[t, forecast_col] = np.sqrt(max(float(sigma2_h), 0.0)) / scale

    return df


def add_egarch_origin_forecast(
    df,
    split_date="2018-01-01",
    h=21,
    scale=100.0,
    forecast_col=None,
    mean="Zero",
    dist="normal",
):
    """
    Creates h-step-ahead EGARCH forecast made at time t and stores it at row t.
    Parameters are estimated only on the training sample, then applied
    recursively over the full dataset.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    split_ts = pd.Timestamp(split_date)

    if forecast_col is None:
        forecast_col = f"EGARCH_Origin_Forecast_h{h}"

    r = (df["LogReturn"] * scale).astype(float)
    train_mask = df["Date"] < split_ts
    returns_train = r.loc[train_mask].dropna()

    df[forecast_col] = np.nan

    if len(returns_train) < 50:
        return df

    model = arch_model(
        returns_train,
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        mean=mean,
        dist=dist,
    )
    res = model.fit(disp="off")

    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    gamma = float(params["gamma[1]"])
    beta = float(params["beta[1]"])
    mu = float(params["mu"]) if "mu" in params.index else 0.0

    n = len(df)

    # container for log-variance over full sample
    log_sigma2 = np.full(n, np.nan)

    # use fitted in-sample conditional variance for training part
    cond_vol_train = res.conditional_volatility
    for pos in returns_train.index.to_numpy():
        sigma_t = float(cond_vol_train.loc[pos])
        log_sigma2[int(pos)] = np.log(max(sigma_t * sigma_t, 1e-18))

    last_train_pos = int(returns_train.index.max())

    # approximation for E|z| under standard normal
    ez_abs = np.sqrt(2.0 / np.pi)

    # continue recursion through full sample using fixed train parameters
    for t in range(last_train_pos + 1, n):
        prev_log_sigma2 = log_sigma2[t - 1]

        if np.isnan(prev_log_sigma2):
            prev_log_sigma2 = np.log(max(np.var(returns_train), 1e-18))

        prev_sigma = np.sqrt(np.exp(prev_log_sigma2))
        prev_eps = float(r.iloc[t - 1] - mu)
        z_prev = prev_eps / max(prev_sigma, 1e-12)

        log_sigma2[t] = (
            omega
            + beta * prev_log_sigma2
            + alpha * (abs(z_prev) - ez_abs)
            + gamma * z_prev
        )

    # h-step-ahead forecast at each origin t, using multi-step egarch formular
    for t in range(n - h):
        log_sigma2_t = log_sigma2[t]
        if np.isnan(log_sigma2_t):
            continue

        if abs(1.0 - beta) < 1e-12:
            log_sigma2_h = log_sigma2_t + h * omega
        else:
            log_sigma2_h = omega * (1.0 - beta**h) / (1.0 - beta) + (beta**h) * log_sigma2_t

        sigma_h = np.sqrt(np.exp(log_sigma2_h)) / scale
        df.at[t, forecast_col] = sigma_h

    return df


def build_target_dated_benchmark_series(df, origin_col, h, out_col):
    """
    Converts an origin-dated forecast column into a target-dated forecast column
    for evaluation/plotting:
        forecast made at t for t+h gets written to row t+h
    """
    df = df.copy()
    df[out_col] = df[origin_col].shift(h)
    return df


# =========================================================
# Feature engineering
# =========================================================

def add_features(df, vstoxx_window=5):
    """
    Create the baseline freature set used by the LSTM models.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # input features
    df["HV_30d"] = df["HV_30d"].clip(lower=vol_floor)
    df["log_HV_30d"] = np.log(df["HV_30d"])
    df["HV_30d_lag1"] = df["HV_30d"].shift(1)
    df["log_HV_30d_lag1"] = df["log_HV_30d"].shift(1)
    df["HV_30d_ma5"] = df["HV_30d"].rolling(window=5, min_periods=1).mean()
    df["log_HV_30d_ma5"] = np.log(df["HV_30d_ma5"].clip(lower=vol_floor))
    df["abs_LogReturn"] = df["LogReturn"].abs()
    df["sq_LogReturn"] = df["LogReturn"] ** 2
    df["rv_5d"] = df["sq_LogReturn"].rolling(window=5, min_periods=1).mean()
    df["rv_10d"] = df["sq_LogReturn"].rolling(window=10, min_periods=1).mean()
    df["log_rv_5d"] = np.log(df["rv_5d"].clip(lower=vol_floor))
    df["log_rv_10d"] = np.log(df["rv_10d"].clip(lower=vol_floor))


    return df

def add_hybrid_log_features(df, h):
    """
    Add log-transformed benchmark forecasts as hybrid input features for the LSTM
    """
    df = df.copy()

    g_col = f"GARCH_Origin_Forecast_h{h}"
    e_col = f"EGARCH_Origin_Forecast_h{h}"

    if g_col in df.columns:
        df[f"log_GARCH_feat_h{h}"] = np.log(df[g_col].clip(lower=vol_floor))
    else:
        df[f"log_GARCH_feat_h{h}"] = np.nan

    if e_col in df.columns:
        df[f"log_EGARCH_feat_h{h}"] = np.log(df[e_col].clip(lower=vol_floor))
    else:
        df[f"log_EGARCH_feat_h{h}"] = np.nan

    return df

def build_model_frame(df, feature_cols, target_col):
    """
    Keep only the columns required for modeling and remove rows with missing values
    introduced by lagging, rolling windows or benchmark features
    """
    cols = ["Date"] + feature_cols
    if target_col not in cols:
        cols.append(target_col)

    return df[cols].dropna().reset_index(drop=True)


# -----------------------------
# Sequence builder
# -----------------------------

def make_sequences(df, feature_cols, target_col, look, h):
    """
    Direct h-step forecasting:
    use rows [i-look+1, ..., i] as input
    and predict row i+h
    """
    values = df.copy().reset_index(drop=True)

    X, y, target_dates = [], [], []

    for i in range(look - 1, len(values) - h):
        x_window = values.iloc[i - look + 1:i + 1][feature_cols].to_numpy(dtype=float)
        y_target = float(values.at[i + h, target_col])
        d_target = pd.to_datetime(values.at[i + h, "Date"])

        X.append(x_window)
        y.append(y_target)
        target_dates.append(d_target)

    return np.array(X), np.array(y), pd.to_datetime(target_dates)

# -----------------------------
# Model
# -----------------------------

class LSTMForecaster(nn.Module):
    """
    LSTM forecaster with a skip-style connection from the last input vector.
    Architecture:
    - one LSTM layer over full sample
    - concatenate last hidden state with last input vector
    - pass through a small fully connected head
    """
    def __init__(self, n_features, hidden_size=32, fc_size=16, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size + n_features, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x):
        # run foll sequence through the LSTM
        out, _ = self.lstm(x)

        # use the final hidden representation and the final observed input
        last_hidden = out[:, -1, :]
        last_input = x[:, -1, :]

        # concatenate both pieces of information before output head
        z = torch.cat([self.dropout(last_hidden), last_input], dim=1)
        z = self.relu(self.fc1(z))
        y = self.fc2(z)
        return y


def fit_predict_lstm(
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
    hidden_size=32,
    fc_size=16,
    dropout=0.1,
    loss_name="mse",
    backtransform_from_log=True,
):
    """
    Train the LSTM on the training sample and generate forecasts for the 
    out-of-sample target dates
    """

    X, y, y_dates = make_sequences(
        df=df_model,
        feature_cols=feature_cols,
        target_col=target_col,
        look=look,
        h=h,
    )

    # split by target date, not by sequence end date
    train_mask = y_dates < pd.Timestamp(split_date)
    test_mask = y_dates >= pd.Timestamp(split_date)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_dates = y_dates[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Empty train or test set after sequence construction.")

    # use final 20% of training period as a time-ordered validation block
    n_total_train = len(X_train)
    n_train_core = int(np.floor(0.8 * n_total_train))

    if n_train_core <= 0 or n_train_core >= n_total_train:
        raise ValueError("Training split too small for time-ordered validation.")

    X_tr = X_train[:n_train_core]
    y_tr = y_train[:n_train_core]
    X_val = X_train[n_train_core:]
    y_val = y_train[n_train_core:]

    # scale using only training core
    scaler = StandardScaler()
    n_features = X_tr.shape[-1]

    # StandardScaler expects 2D input, so reshape sequences
    X_tr_2d = X_tr.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler.fit(X_tr_2d)

    # reshape back into 3D
    X_tr = scaler.transform(X_tr_2d).reshape(-1, look, n_features)
    X_val = scaler.transform(X_val_2d).reshape(-1, look, n_features)
    X_test = scaler.transform(X_test_2d).reshape(-1, look, n_features)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # convert arrays to tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # mini-batch loader, shuffle disabled to preserve temporal order
    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=False,
    )

    # initialize model, loss and optimizer
    model = LSTMForecaster(
        n_features=n_features,
        hidden_size=hidden_size,
        fc_size=fc_size,
        dropout=dropout,
    ).to(device)

    criterion = get_loss(loss_name)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # early stopping params
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    # ---------
    # TRAINING loop
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_train_loss = 0.0

        # batch
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            # gradient clipping helps stabilize the LSTM training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * xb.size(0)

        train_loss = running_train_loss / max(1, len(X_tr_t))

        # validation pass
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device))
            val_loss = criterion(val_pred, y_val_t.to(device)).item()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        # keep best model according to validation loss
        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # generate test-set forecasts
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t.to(device)).detach().cpu().numpy().reshape(-1)

    # backtransform from log-vola to level vola
    if backtransform_from_log:
        y_pred_level = np.exp(y_pred)
        y_test_level = np.exp(y_test)
        y_train_level = np.exp(y_train)
    else:
        y_pred_level = y_pred.copy()
        y_test_level = y_test.copy()
        y_train_level = y_train.copy()

    # enforce positivity and cap extreme prediction based on training
    y_pred_level = np.clip(y_pred_level, vol_floor, None)
    soft_cap = float(np.quantile(y_train_level, 0.9995))
    y_pred_level = np.clip(y_pred_level, vol_floor, soft_cap)

    out = pd.DataFrame({
        "Date": test_dates,
        "Observed_HV_30d": y_test_level,
        "Forecast": y_pred_level,
    })

    return out



#############################################################################################################
## MAIN
#############################################################################################################

def main():

    """
    1. load and preprocess the data.
    2. create baseline features.
    3. estimate split GARCH and EGARCH 
    4. build plain and hybrid feature sets.
    5. train plain, GARCH-hybrid, and EGARCH-hybrid LSTM models.
    6. merge outputs, evaluate forecasts, and plot the results.
    """

    global asset_name, data_path, split_date, horizon

    args = parse_args()

    asset_name = args.asset_name
    data_path = args.data_path
    split_date = pd.Timestamp(args.split_date)
    horizon = args.horizon

    # read horizon-specific network
    cfg = get_horizon_config(horizon)
    hidden_size = cfg["hidden_size"]
    fc_size = cfg["fc_size"]
    dropout = cfg["dropout"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    loss_name = cfg["loss_name"]
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

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # -------------------------
    # Base features
    # -------------------------
    df = add_features(df, vstoxx_window=5)

    # -------------------------
    # split garch, egarch
    # -------------------------
    df = add_garch_origin_forecast(
        df=df,
        split_date=split_date,
        h=horizon,
        scale=100.0,
        forecast_col=f"GARCH_Origin_Forecast_h{horizon}",
        mean="Constant",
        dist="normal",
    )

    df = add_egarch_origin_forecast(
        df=df,
        split_date=split_date,
        h=horizon,
        scale=100.0,
        forecast_col=f"EGARCH_Origin_Forecast_h{horizon}",
        mean="Zero",
        dist="normal",
    )

    # add log-transformed origin-dated benchmark forecasts as hybrid features
    df = add_hybrid_log_features(df, h=horizon)

    # target-dated benchmark columns for plotting / comparison
    df = build_target_dated_benchmark_series(
        df=df,
        origin_col=f"GARCH_Origin_Forecast_h{horizon}",
        h=horizon,
        out_col=f"GARCH_Split_Forecast_h{horizon}",
    )

    df = build_target_dated_benchmark_series(
        df=df,
        origin_col=f"EGARCH_Origin_Forecast_h{horizon}",
        h=horizon,
        out_col=f"EGARCH_Split_Forecast_h{horizon}",
    )

    # additional input features
    df["rv_21d"] = df["sq_LogReturn"].rolling(window=21, min_periods=1).mean()
    df["rv_63d"] = df["sq_LogReturn"].rolling(window=63, min_periods=1).mean()

    df["log_rv_21d"] = np.log(df["rv_21d"].clip(lower=vol_floor))
    df["log_rv_63d"] = np.log(df["rv_63d"].clip(lower=vol_floor))

    neg_ret = df["LogReturn"].clip(upper=0.0)
    df["downside_rv_21d"] = (neg_ret ** 2).rolling(window=21, min_periods=1).mean()
    df["log_downside_rv_21d"] = np.log(df["downside_rv_21d"].clip(lower=vol_floor))

    df["ewma_var"] = df["sq_LogReturn"].ewm(alpha=1 - 0.94, adjust=False).mean()
    df["log_ewma_vol"] = 0.5 * np.log(df["ewma_var"].clip(lower=vol_floor))
    # -------------------------
    # Feature sets
    # -------------------------
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

     # hybrid feature sets: baseline features plus one econometric feature
    ghyb_cols = plain_feature_cols + [f"log_GARCH_feat_h{horizon}"]
    ehyb_cols = plain_feature_cols + [f"log_EGARCH_feat_h{horizon}"]

    # -------------------------
    # Model frames
    # -------------------------

    # remove rows with missing values separately for each feature specification
    df_model_plain = build_model_frame(df, plain_feature_cols, target_col)
    df_model_g = build_model_frame(df, ghyb_cols, target_col)
    df_model_e = build_model_frame(df, ehyb_cols, target_col)

    # -------------------------
    # Train + predict
    # -------------------------
    # plain LSTM
    out_plain = fit_predict_lstm(
        df_model=df_model_plain,
        feature_cols=plain_feature_cols,
        target_col=target_col,
        look=look,
        split_date=split_date,
        h=horizon,
        seed=seed,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        hidden_size=hidden_size,
        fc_size=fc_size,
        dropout=dropout,
        loss_name=loss_name,
        backtransform_from_log=use_log_target,
    ).rename(columns={"Forecast": f"LSTM_Forecast_h{horizon}"})

    # G-LSTM
    out_g = fit_predict_lstm(
        df_model=df_model_g,
        feature_cols=ghyb_cols,
        target_col=target_col,
        look=look,
        split_date=split_date,
        h=horizon,
        seed=seed,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        hidden_size=hidden_size,
        fc_size=fc_size,
        dropout=dropout,
        loss_name=loss_name,
        backtransform_from_log=use_log_target,
    ).rename(columns={"Forecast": f"G-LSTM_Forecast_h{horizon}"})

    # E-LSTM
    out_e = fit_predict_lstm(
        df_model=df_model_e,
        feature_cols=ehyb_cols,
        target_col=target_col,
        look=look,
        split_date=split_date,
        h=horizon,
        seed=seed,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        hidden_size=hidden_size,
        fc_size=fc_size,
        dropout=dropout,
        loss_name=loss_name,
        backtransform_from_log=use_log_target,
    ).rename(columns={"Forecast": f"E-LSTM_Forecast_h{horizon}"})

    # -------------------------
    # Merge outputs on target date
    # -------------------------
    # start from plain LSTM and use observed vola as reference
    out = out_plain[["Date", "Observed_HV_30d", f"LSTM_Forecast_h{horizon}"]].copy()
    out = out.rename(columns={"Observed_HV_30d": "HV_30d"})

    # add hybrid forecasts
    out = out.merge(
        out_g[["Date", f"G-LSTM_Forecast_h{horizon}"]],
        on="Date",
        how="left",
    )
    out = out.merge(
        out_e[["Date", f"E-LSTM_Forecast_h{horizon}"]],
        on="Date",
        how="left",
    )

    out = out.merge(
        df[[
            "Date",
            f"GARCH_Split_Forecast_h{horizon}",
            f"EGARCH_Split_Forecast_h{horizon}",
        ]],
        on="Date",
        how="left",
    )

   

    # -------------------------
    # Save
    # -------------------------
    forecast_path = os.path.join(
        output_dir,
        f"lstm_hybrid_forecasts_h{horizon}.csv",
    )

    plot_path = os.path.join(
        output_dir,
        f"{asset_name}_lstm_h{horizon}.png",
    )

    #out.to_csv(forecast_path, index=False, sep=";")
    #eval_df.to_csv(eval_path, index=False, sep=";")

    #print(f"\nWrote forecast file: {forecast_path}")
    #print(f"Wrote evaluation file: {eval_path}")

    # -------------------------
    # Plot
    # -------------------------

    forecast_specs = [
        (f"LSTM_Forecast_h{horizon}", "LSTM"),
        (f"G-LSTM_Forecast_h{horizon}", "G-LSTM"),
        (f"E-LSTM_Forecast_h{horizon}", "E-LSTM"),
    ]

    plot_volatility_forecasts(
        df=out,
        forecast_specs=forecast_specs,
        asset_name=asset_name,
        model_name="LSTM Models",
        horizon=horizon,
        split_date=split_date,
        oos_only=True,
        save_path="plots/eux_mitvstoxx_lstm_h21",
        show=True,
    )


if __name__ == "__main__":
    main()


# USAGE OF ARGS# The args are optional. If no args are passed, the script uses
# the default values defined in parse_args():
#
#   --horizon     default: 63
#   --asset-name  default: "USD/EUR"
#   --data-path   default: "USD_EUR Historical Data.csv"
#   --split-date  default: "2018-01-01"
#   --use-vstoxx  default: False
#   --vstoxx-path default: "VSTOXX EUR History.csv"
#
# ---------------------------------------------------------
# 1) Start with all defaults
# ---------------------------------------------------------
# Example:
#   python my_lstm_script.py
#
# Then the script runs with:
#   horizon    = 63
#   asset_name = "USD/EUR"
#   data_path  = "USD_EUR Historical Data.csv"
#   split_date = "2018-01-01"
#   use_vstoxx = False
#
#
# ---------------------------------------------------------
# 2) Start with a specific horizon
# ---------------------------------------------------------
# Example:
#   python my_lstm_script.py --horizon 21
#
# or:
#   python my_lstm_script.py --horizon 63
#
# The horizon also determines which hyperparameters are chosen in
# get_horizon_config(horizon).
#
#
# ---------------------------------------------------------
# 3) Switch to another asset / another input file
# ---------------------------------------------------------
# Example for BMW:
#   python my_lstm_script.py --asset-name "BMW" --data-path "BMW Historical Data.csv" --horizon 21
#
# Example for DAX:
#   python my_lstm_script.py --asset-name "DAX" --data-path "DAX Historical Data.csv" --horizon 63
#
# Example for EURO STOXX:
#   python my_lstm_script.py --asset-name "EURO STOXX 50" --data-path "Euro_Stoxx_50 Historical Data.csv" --horizon 21
#
# Important:
#   --asset-name is only the label used in the plot title.
#   --data-path is the actual CSV file the script reads.
#
#
# ---------------------------------------------------------
# 4) Use VSTOXX as additional input feature
# ---------------------------------------------------------
# If I want to include VSTOXX, I must explicitly add the flag
# --use-vstoxx.
#
# Example:
#   python my_lstm_script.py --horizon 21 --use-vstoxx --vstoxx-path "VSTOXX EUR History.csv"
#
# If I do NOT write --use-vstoxx, then use_vstoxx stays False,
# even if a vstoxx path is given.
#
#
# ---------------------------------------------------------
# 5) Change the split date
# ---------------------------------------------------------
# Example:
#   python my_lstm_script.py --horizon 21 --split-date 2019-01-01
#
#