import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from plot_forecasts import plot_volatility_forecasts

from econometric_models import load_and_prepare_data

##################################################################
# Sript for the MLP model forecast for the one-step-ahead forecast
##################################################################



# -----------------------------------------------------------------
# Lagged Feature Construction
# -----------------------------------------------------------------
def make_lagged_matrix(df, hv_col="HV_30d", p=30, use_vdax=True, vdax_col="VDAX"):
    """
    Transform time series of historical volatility into a supervised learning dataset.
    Input in function: dataset, historical volatility, p (number of lags)
    Return: tuple X,y 
     - X: input features
     - y: target values
    """
    data = pd.DataFrame(index=df.index)

    # create lagged input feature
    # lag=1: HV_30d_lag1: df[hv_col].shift(1) -> yesterdays vola
    # lag=2: HV_30d_lag2: df[hv_col].shift(2) -> the vola from two days ago
    for lag in range(1, p + 1):
        data[f"{hv_col}_lag{lag}"] = df[hv_col].shift(lag)
    
    # target variable: next days vola
    y = df[hv_col].shift(-1)

    feature_cols = [f"{hv_col}_lag{lag}" for lag in range(1, p + 1)]

    # remove rows where lagged inputs are missing
    data = data.dropna(subset=feature_cols)
    y = y.loc[data.index]

    # keep only rows where target exists
    mask = y.notna()
    data = data.loc[mask]
    y = y.loc[mask]

    return data, y


# -----------------------------
# 2. Define MLP model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(32, 16)):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # add hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))       # linear transformation
            layers.append(nn.ReLU())                    # activation functions
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))  # final layer: gives only one output value (forecast)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # pass input through all layers
        return self.network(x)


# -----------------------------
# 3. Training + Forecasting
# -----------------------------
def add_mlp_forecast(
    df,
    hv_col="HV_30d",
    p=30,
    split_date="2018-01-01",
    hidden_dims=(32, 16),
    lr=1e-3,
    batch_size=64,
    epochs=2000,
    patience=25,
    weight_decay=1e-3,
    seed=42,
):

    """
    Trains the MLP and adds one-step-ahead forecasts to the dataframe. 
    """
    # fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = df.copy()
    X, y = make_lagged_matrix(df, hv_col, p)

    split_ts = pd.Timestamp(split_date)

    # X is indexed by time t, but y belongs to t+1
    # shift index forward to match the forecast date
    origin_pos = df.index.get_indexer(X.index)
    target_pos = origin_pos + 1

    # keep only rows where t+1 still exists in dataset
    valid = target_pos < len(df)

    X = X.iloc[valid]
    y = y.iloc[valid]
    target_idx = df.index[target_pos[valid]]
    
    # rchange index of X and y to the actual forecasted date (t+1)
    X = X.set_axis(target_idx)
    y = y.set_axis(target_idx)

    # use date column to spliti into train and test sample
    target_dates = df.loc[target_idx, "Date"]
    train_mask = target_dates < split_ts
    test_mask = target_dates >= split_ts

    # standardize inputs using only training data 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.loc[train_mask])
    X_test = scaler.transform(X.loc[test_mask])

    y_train = y.loc[train_mask].values.reshape(-1, 1)
    y_test = y.loc[test_mask].values.reshape(-1, 1)

    # Convert training data to pytorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # create batches for training
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # build the MLP model
    model = MLP(input_dim=X.shape[1], hidden_dims=hidden_dims)

    # use mean squared error as the loss function
    criterion = nn.MSELoss()

    # adam optimizer for parameter udpate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # variables for early stopping
    best_loss = np.inf
    counter = 0

    # training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # go throuhg all batches
        for xb, yb in loader:
            optimizer.zero_grad()       # clear old gradients
            pred = model(xb)            # make predictions
            loss = criterion(pred, yb)  # compare prediction and true value
            loss.backward()             # compute gradients
            optimizer.step()            # update weights
            epoch_loss += loss.item()

        # average loss over all batches
        epoch_loss /= len(loader)

    # save model if improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        # stop training if there is no improvement for 'patience' epochs
        if counter >= patience:
            break

    # restore best model during training
    model.load_state_dict(best_state)
    model.eval()

    # create forecasts for all available observations
    X_all = scaler.transform(X)
    X_all = torch.tensor(X_all, dtype=torch.float32)

    # small lower bound -> forecast cannot become negative or zero
    VOL_FLOOR = 1e-6

    with torch.no_grad():
        yhat = model(X_all).numpy().flatten()
        yhat = np.clip(yhat, VOL_FLOOR, None)

    df["MLP_Forecast"] = np.nan
    df.loc[target_idx, "MLP_Forecast"] = yhat  # write forecast to date t+1

    return df


#############################################################################################################
## MAIN
#############################################################################################################

if __name__ == "__main__":

    split_date = pd.Timestamp("2018-01-01")
    
    # load and prepare data
    df = load_and_prepare_data(
        path = "DAX Historical Data.csv",
        sep=";",
        decimal=",",
        date_format="%m.%d.%Y",
        drop_cols=["Open", "High", "Low", "Vol,", "Change %"]
    )

    # train model and add forecasts
    df = add_mlp_forecast(
        df=df,
        hv_col="HV_30d",
        p=30,
        split_date=split_date,
        hidden_dims=(32,16),
        lr=1e-3
    )

    # keep only the relevant columns for output
    out_cols = ["Date", "HV_30d", "MLP_Forecast"]

    # remove rows where true HV or forecast is missing
    df_out = df[out_cols].dropna(subset=["HV_30d", "MLP_Forecast"])

    # mark each row as train and test sample
    df_out["Sample"] = np.where( df_out["Date"] < split_date, "Train", "Test")

    # save results to csv
    df_out.to_csv("outputs/mlp_forecasts.csv", index=False, sep=";")
    print("mlp_forecast.csv got created.")

    df =  df_out.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # keep only out-of-sample period for plotting
    df_oos = df[df["Date"] >= split_date]

    plot_volatility_forecasts(
    df=df_oos,
    forecast_specs=[
        ("MLP_Forecast", "MLP"),
    ],
    asset_name="DAX",
    model_name="MLP",
    horizon=1,
    oos_only=True,
    split_date="2018-01-01",
    save_path="plots/dax_mlp.png",
    show=True
)

    