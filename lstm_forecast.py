import numpy as np
import pandas as pd
from arch import arch_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from econometric_models import load_and_prepare_data
from plot_forecasts import plot_volatility_forecasts

##################################################################
# Sript for the LSTM model forecast for the one-step-ahead forecast
##################################################################


# -----------------------------------------------------------------
# For the hybrid model: GARCH forecasts
# -----------------------------------------------------------------

def add_split_garch_forecast(
        df,
        split_date,
        scale=100.0,
        forecast_col="Split_GARCH_Forecast"
    ):
    """
    Fits a GARCH(1,1) model only on train data and creates then one-step-ahead
    forecasts for the full sample.
    The train-estimated parameters are also used to create forecasts in the train period,
    because LSTM later needs these values as input features.
    """

    df = df.copy()

    # make sure the date column is in datetime format
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    # sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    split_date = pd.Timestamp(split_date)

    # split data into train and test
    train_mask = df["Date"] < split_date
    test_mask  = df["Date"] >= split_date

    # scale returns
    returns_scaled = df["LogReturn"] * scale

    # use only train returns before fitting the GARCH model
    returns_train = returns_scaled[train_mask]

    # fit GARCH(1,1) on train period
    garch = arch_model(
        returns_train,
        vol="GARCH",
        p=1,
        q=1,
        mean="Constant",
        dist="normal"
    )

    # estimate model parameters and get them
    res = garch.fit(disp="off")
    params = res.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta  = params["beta[1]"]
    mu    = params["mu"]   

    # last train day t0 before the split

    last_train_idx = returns_train.index[-1]

    # Bedingte Volatilität zum letzten Train-Zeitpunkt (skaliert)
    sigma_t = res.conditional_volatility.loc[last_train_idx] 
    sigma2_t = sigma_t ** 2

    # eps_t = r_t - m
    # residual at last train day
    eps_t = returns_scaled.loc[last_train_idx] - mu

    # create empty forecast series
    garch_forecast = pd.Series(np.nan, index=df.index, dtype=float)

    # -------------------------
    # TRAIN PART: in-sample 1-step-ahead feature
    
    # convert conditional volatility back to original scale
    train_condvol = (res.conditional_volatility / scale)

    # shift by one day, so value at t is the forecast made at t-1 for t
    garch_forecast.loc[train_condvol.index] = train_condvol.shift(1)

    # -------------------------
    # TEST PART: out-of-sample recursive 1-step-ahead

    full_index = df.index
    pos_last_train = full_index.get_loc(last_train_idx)

    for pos in range(pos_last_train + 1, len(full_index)):
        idx = full_index[pos]

        # standard GARCH(1,1) variance recursion
        sigma2_next = omega + alpha * (eps_t ** 2) + beta * sigma2_t
        sigma_next  = np.sqrt(sigma2_next)

        # again back into original scale
        garch_forecast.iloc[pos] = sigma_next / scale

        # update values for the next step
        eps_t    = returns_scaled.loc[idx] - mu
        sigma2_t = sigma2_next

    df[forecast_col] = garch_forecast
    # forecast_col has values for train and test set with only train parameters
    return df

# -----------------------------------------------------------------
# For the hybrid model: EGARCH forecasts
# -----------------------------------------------------------------

def add_split_egarch_forecast(
        df,
        split_date,
        scale=100.0,
        forecast_col="Split_EGARCH_Forecast"
    ):
    """
    Fits a EGARCH(1,1) model only on train data and creates then one-step-ahead
    forecasts for the full sample.
    The train-estimated parameters are also used to create forecasts in the train period,
    because LSTM later needs these values as input features.
    """
    df = df.copy()

    # make sure date is datetime
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)
    split_date = pd.Timestamp(split_date)

    # training sample
    train_mask = df["Date"] < split_date
    returns_scaled = df["LogReturn"] * scale
    returns_train = returns_scaled[train_mask]

    # fit EGARCH(1,1) on training days
    egarch = arch_model(
        returns_train,
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        mean="Constant",
        dist="normal"
    )

    # get estimated estimated parameters
    res = egarch.fit(disp="off")
    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    gamma = float(params["gamma[1]"])
    beta  = float(params["beta[1]"])
    mu    = float(params["mu"])

    #z is standardized shock and normal distributed
    E_abs_z = np.sqrt(2.0 / np.pi)

    last_train_idx = returns_train.index[-1] # last train day

    # starting values from last training day
    sigma_t = float(res.conditional_volatility.loc[last_train_idx])
    sigma2_t = sigma_t ** 2
    log_sigma2_t = np.log(sigma2_t)
    eps_t = float(returns_scaled.loc[last_train_idx] - mu)

    # empty forecast series
    egarch_forecast = pd.Series(np.nan, index=df.index, dtype=float)

    # Train: in-sample 1-step-ahead feature
    train_condvol = (res.conditional_volatility / scale)
    egarch_forecast.loc[train_condvol.index] = train_condvol.shift(1)

    # Test: recursive 1-step-ahead
    full_index = df.index
    pos_last_train = full_index.get_loc(last_train_idx)

    for pos in range(pos_last_train + 1, len(full_index)):
        idx = full_index[pos]

        # standardized shock
        z_t = eps_t / sigma_t if sigma_t > 0 else 0.0

        # EGARCH recursion in log-variance form
        log_sigma2_next = (
            omega
            + beta * log_sigma2_t
            + alpha * (np.abs(z_t) - E_abs_z)
            + gamma * z_t
        )

        sigma2_next = float(np.exp(log_sigma2_next))
        sigma_next  = float(np.sqrt(sigma2_next))

        # back to original scale
        egarch_forecast.iloc[pos] = sigma_next / scale

        # upate valus for the next step
        eps_t = float(returns_scaled.loc[idx] - mu)
        sigma_t = sigma_next
        log_sigma2_t = log_sigma2_next

    df[forecast_col] = egarch_forecast
    return df

# -----------------------------------------------------------------
# Lagged Feature Construction
# -----------------------------------------------------------------

def make_sequences(df, feature_cols, target_col, lookback):
    """
    Similar to the MLP case: turns time series into sequences for LSTM.
    """
    X, y, dates = [], [], []

    X_df = df[feature_cols].copy()
    y_s  = df[target_col].copy()

    for i in range(lookback, len(df) - 1):
        # input sequence: from i-lookback up to i
        X.append(X_df.iloc[i - lookback:i+1].to_numpy())   

        # target: next days volatility     
        y.append(y_s.iloc[i + 1])

        # store the date of the target
        dates.append(df.iloc[i + 1]["Date"])

    return np.array(X), np.array(y, dtype=float), pd.to_datetime(dates)

# -----------------------------------------------------------------
# LSTM construction
# -----------------------------------------------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, fc_size: int = 16):
        super().__init__()
        # LSTM layer processe the time sequence
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # fully connected layers turn the LSTM output into one forecast value
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x):
        # x has shape: (batch, sequence length, number of features)
        out, _ = self.lstm(x)          
        
        # use only the output from last time step
        last = out[:, -1, :]           
        
        # feed it tthrough the dense layers
        z = self.fc1(last)            
        z = self.relu(z)
        y = self.fc2(z)  

        # softplus makes sure that the forecast stays positive             
        y = torch.nn.functional.softplus(y) + 1e-6
        return y   

def _set_seed(seed: int = 42):
    """
    Fix random seed for reproducible results
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------
# Extracts GARCH and EGARCH parameters
# -----------------------------------------------------------------

def extract_garch_params(params: pd.Series):
    """
    Extracts (mu, omega, alpha, beta) from a fitted model result.
    Different versions of arch_model may use slighly different parameter names,
    so this function checks several possible names.
    """

    # mean parameter
    mu = 0.0
    for k in ["mu", "Const", "const", "Mean", "mean"]:
        if k in params.index:
            mu = float(params[k])
            break

    # variance parameter
    omega = float(params["omega"]) if "omega" in params.index else None  
    
    # alpha, beta parameter
    alpha = None
    beta = None

    for k in ["alpha[1]", "alpha[0]", "alpha1", "alpha"]:
        if k in params.index:
            alpha = float(params[k])
            break

    for k in ["beta[1]", "beta[0]", "beta1", "beta"]:
        if k in params.index:
            beta = float(params[k])
            break
    
    if omega is None or alpha is None or beta is None:
        raise ValueError(f"Could not parse GARCH params from: {list(params.index)}")
    
    return mu, omega, alpha, beta

# -----------------------------------------------------------------
# LSTM training and forecasting
# -----------------------------------------------------------------

def fit_predict_lstm(
    df_model,
    feature_cols,
    target_col,
    lookback,
    split_date,
    seed=42,
    batch_size=32,
    max_epochs=200,
    patience=10,
    lr=1e-3,
):
    """
    fits the LSTM model and returns forecasts for the test period.
    """

    # build sequences
    X, y, y_dates = make_sequences(df_model, feature_cols, target_col, lookback)

    # ensure y is one-dimensional
    if y.ndim == 2:
        # take first (or the correct) column
        y = y[:, 0]

    # split into train and test using target dates
    train_mask = y_dates < split_date
    test_mask = y_dates >= split_date

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_dates = y_dates[test_mask]

    # scale features using train only -> each feature gets transformed to:
    # mean = 0, standard deviation = 1
    scaler = StandardScaler()
    n_features = X_train.shape[-1]

    # flatten sequences to apply scaler (from 3 dimensions into 2)
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler.fit(X_train_2d) # fit scalar only on train data

    # transform and reshape back to sequence format (into 3D)
    seq_len = X_train.shape[1]
    X_train = scaler.transform(X_train_2d).reshape(-1, seq_len, n_features)
    X_test = scaler.transform(X_test_2d).reshape(-1, seq_len, n_features)

    # set seeds for reproducability
    _set_seed(seed)
    np.random.seed(seed)

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create explicit validation split (last 20% of train)
    n = len(X_train)
    n_train = int(np.floor(0.8 * n))
    X_tr, y_tr = X_train[:n_train], y_train[:n_train]
    X_val, y_val = X_train[n_train:], y_train[n_train:]

    # convert data to pytorch tensors (works not with NumPy arrays)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).unsqueeze(1) 
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)   
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # create dataloader, it splits the training set into mini-bathces
    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=False  # since it istime series data
    )

    # create model, loss function and optimizer
    # LSTMForecaster defined befor, processes each sequence and outputs one forecast value
    model = LSTMForecaster(n_features=n_features, hidden_size=32, fc_size=16).to(device)
    
    # loss again mean squared error and adam updates parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # early stopping preparation
    best_val = float("inf")     # stores lowest validation
    best_state = None           # stores corresponding weights
    bad_epochs = 0              # counts how many epochs passed without improvement

    # ----------------------------
    # TRAINing loop
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0

        # go through all mini-batches of training set
        for xb, yb in train_loader:
            # move current batch to GPU or CPU
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()                   # clear old gradients from previous step
            pred = model(xb)                        # forward pass: model produces preditons
            loss = criterion(pred, yb)              # compute loss
            loss.backward()                         # packward pass: compute gradients 
            optimizer.step()                        # update model parameters using adam
            running += loss.item() * xb.size(0)     # add batch loss to running sum, xb.size is batch size

        # average training loss (over all training samples
        train_loss = running / max(1, len(X_tr_t))

        # validation setup
        model.eval()

        # no gradients are needed during validation
        with torch.no_grad():
            val_pred = model(X_val_t.to(device))
            val_loss = criterion(val_pred, y_val_t.to(device)).item()

        # print progress
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # Early stopping logic: if val. loss improved -> save current model, otherwise
        # increase counter
        if val_loss < best_val - 1e-12:
            best_val = val_loss
            # save a copy of current best model weights
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.6f})")
                break

    # restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # predict on test data: model now creates forecasts for the unseen test period
    VOL_FLOOR = 1e-6

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t.to(device)).detach().cpu().numpy().reshape(-1)

        # convert explicitly to float array
        y_pred = np.asarray(y_pred, dtype=float)

        # replace invalid values by NaN
        y_pred[~np.isfinite(y_pred)] = np.nan

        # ensure forecast not negative
        y_pred = np.clip(y_pred, VOL_FLOOR, None)

    # return prediction together with their dates
    out = pd.DataFrame({
        "Date": test_dates,
        f"{target_col}_pred": y_pred
    })
    return out

def worst_jump_days(out, col, top=25, eps=1e-12):
    """
    Find days with biggest forecast jumps
    """ 

    d = out[["Date", col, "HV_30d"]].copy()
    d["Date"] = pd.to_datetime(d["Date"])
    x = pd.to_numeric(d[col], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    d[col] = x

    d["abs_jump"] = d[col].diff().abs()
    d["rel_jump"] = d["abs_jump"] / (d[col].shift(1).abs() + eps)

    return d.sort_values("rel_jump", ascending=False).head(top)

#############################################################################################################
## MAIN
#############################################################################################################

def main():
    split_date = pd.Timestamp("2018-01-01")
    lookback = 60
    target_col = "HV_30d"

    df = load_and_prepare_data(
        path = "Euro Stoxx 50 Historical Data.csv",
        sep=";",
        decimal=",",
        date_format="%m.%d.%Y",
        drop_cols=["Open", "High", "Low", "Vol,", "Change %"]
    )

    # ensure sorted dates
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # basic input features for plain LSTM
    base_cols = ["LogReturn", "HV_30d"]

    df_base = df[["Date"] + base_cols].copy()
    
    # remove rows with missing values
    df_model_1 = df[["Date"] + base_cols + [target_col]].dropna().reset_index(drop=True)

    # plain LSTM forecast
    out_lstm = fit_predict_lstm(
        df_model=df_model_1,
        feature_cols=base_cols,
        target_col=target_col,
        lookback=lookback,
        split_date=split_date,
        seed=42
    ).rename(columns={f"{target_col}_pred": "LSTM_Forecast"})

    # -----------------------------------------------------------------
    # Hybrid models
    # -----------------------------------------------------------------

    df_g = add_split_garch_forecast(
        df,
        split_date=split_date,
        scale=100,
        forecast_col="Split_GARCH_Forecast"
    )
    
    df_e = add_split_egarch_forecast(
        df,
        split_date=split_date,
        scale=100,
        forecast_col="Split_EGARCH_Forecast"
    )

    df_ge = (df_base.merge(df_g[["Date", "Split_GARCH_Forecast"]], on="Date", how="left")
             .merge(df_e[["Date", "Split_EGARCH_Forecast"]], on="Date", how="left")
             )

    # feature sets for hybrid models
    target_col = "HV_30d"
    feature_cols_ghybrid = ["LogReturn", "HV_30d", "Split_GARCH_Forecast"]
    feature_cols_ehybrid = ["LogReturn", "HV_30d", "Split_EGARCH_Forecast"]
    #feature_cols_gehybrid = ["LogReturn", "HV_30d", "Split_GARCH_Forecast", "Split_EGARCH_Forecast"]

    # clean forecat columns and make sure values are always positive
    if "Split_GARCH_Forecast" in df_g.columns:
        df_g["Split_GARCH_Forecast"] = df_g["Split_GARCH_Forecast"].replace([np.inf, -np.inf], np.nan).clip(lower=1e-6)

    if "Split_EGARCH_Forecast" in df_e.columns:
        df_e["Split_EGARCH_Forecast"] = df_e["Split_EGARCH_Forecast"].replace([np.inf, -np.inf], np.nan).clip(lower=1e-6)

    for c in ["Split_GARCH_Forecast", "Split_EGARCH_Forecast"]:
        if c in df_ge.columns:
            df_ge[c] = df_ge[c].replace([np.inf, -np.inf], np.nan).clip(lower=1e-6)
    
    # prepare datasets for hybrid LSTMs
    df_model_2 = df_g[["Date"] + feature_cols_ghybrid + [target_col]].dropna().reset_index(drop=True)
    df_model_3 = df_e[["Date"] + feature_cols_ehybrid + [target_col]].dropna().reset_index(drop=True)
    #df_model_4 = df_ge[["Date"] + feature_cols_gehybrid + [target_col]].dropna().reset_index(drop=True)

    # G-LSTM forecast
    out_ghybrid = fit_predict_lstm(
        df_model=df_model_2,
        feature_cols=feature_cols_ghybrid,
        target_col=target_col,
        lookback=lookback,
        split_date=split_date,
        seed=42
    ).rename(columns={f"{target_col}_pred": "G-LSTM_Forecast"})

    # E-LSTM forecast
    out_ehybrid = fit_predict_lstm(
        df_model=df_model_3,
        feature_cols=feature_cols_ehybrid,
        target_col=target_col,
        lookback=lookback,
        split_date=split_date,
        seed=42
    ).rename(columns={f"{target_col}_pred": "E-LSTM_Forecast"})

    """ out_gehybrid = fit_predict_lstm(
        df_model=df_model_4,
        feature_cols=feature_cols_gehybrid,
        target_col=target_col,
        lookback=lookback,
        split_date=split_date,
        seed=42
    ).rename(columns={f"{target_col}_pred": "GE-LSTM_Forecast"})"""

    # goal: build one final output table
    # start with plain LSTM output
    out = out_lstm[["Date", "LSTM_Forecast"]].copy()

    # add true observed volatility valeus
    out = out.merge(df[["Date", "HV_30d"]], on="Date", how="left")

    # add parametric forecast series
    out = out.merge(
        df_ge[["Date", "Split_GARCH_Forecast", "Split_EGARCH_Forecast"]],
        on="Date",
        how="left"
    )

    # add hybrid LSTM forecasts
    out = out.merge(out_ghybrid[["Date", "G-LSTM_Forecast"]], on="Date", how="left")
    out = out.merge(out_ehybrid[["Date", "E-LSTM_Forecast"]], on="Date", how="left")
    #out = out.merge(out_gehybrid[["Date", "GE-LSTM_Forecast"]], on="Date", how="left")

    # final output columns
    out = out[[
        "Date",
        "HV_30d",
        "LSTM_Forecast",
        "G-LSTM_Forecast",
        "E-LSTM_Forecast",
    ]]

    #print(worst_jump_days(out, "Roll_EGARCH_Forecast", top=30))
    

    plot_volatility_forecasts(
        df=out,
        forecast_specs=[
            ("LSTM_Forecast", "LSTM"),
            ("G-LSTM_Forecast", "G-LSTM"),
            ("E-LSTM_Forecast", "E-LSTM"),
        ],
        asset_name="Euro Stoxx",
        model_name="LSTM Models",
        split_date="2018-01-01",
        oos_only=True,
        horizon=1,
        save_path="plots/eux2_lstm.png",
        show=True
    )

    forecast_cols = [
    "LSTM_Forecast",
    "G-LSTM_Forecast",
    "E-LSTM_Forecast",
    ]

    # final cleaning setup
    for c in forecast_cols:
        out[c] = (
            out[c]
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=1e-6)
        )
    
    # print share of values at the lower bound
    for c in forecast_cols:
        print(c, (out[c] <= 1e-6).mean())

    # save output to csv file
    #out.to_csv("outputs/lstm_and_splithybrid_forecasts.csv", index=False, sep=";")
    #print("Wrote lstm_and_splithybrid_forecasts.csv with", len(out), "rows.")

   
if __name__ == "__main__":
    main()