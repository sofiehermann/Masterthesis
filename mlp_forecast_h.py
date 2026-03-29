import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from econometric_models_h import vol_floor, load_and_prepare_data, clip_forecast_series
from plot_forecasts import plot_volatility_forecasts

##################################################################
# used script for MLP k-step-ahead forecast (simpler feature set)
##################################################################

# -----------------------------
# 1. Lagged Feature Construction
# -----------------------------
def make_lagged_matrix_h(df, hv_col="HV_30d", p=30, h=21):
    """
    Build lag features up to time t and a target y_t = HV_{t+h}.
    Returns:
      X indexed by origin dates (t),
      y aligned with X index (still indexed by origin dates).
    """
    # start with empty dataframe using same index as df
    data = pd.DataFrame(index=df.index)
    
    # create p lagged volatility features
    for lag in range(1, p + 1):
        data[f"{hv_col}_lag{lag}"] = df[hv_col].shift(lag)

    # target at horizon h
    y = df[hv_col].shift(-h)

    # remove rows with missing feature values caused by lag construction
    data = data.dropna()

    # keep y aligned with remaing valid feature rows
    y = y.loc[data.index]

    # remove rows where target is missing
    mask = y.notna()
    data = data.loc[mask]
    y = y.loc[mask]

    return data, y


# -----------------------------
# 2. Define MLP Architecture
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(32, 16)):
        """
        MLP with ReLU activations.
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # add one linear layer # ReLU activation for each hidden layer
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd

        # final output layer producing a single forecast value
        layers.append(nn.Linear(prev_dim, 1))

        # stack all layers into one sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # forwardpass through the network
        return self.network(x)


# -----------------------------
# 3. Training + Forecasting (h-step)
# -----------------------------
def add_mlp_forecast_h(
    df,
    hv_col="HV_30d",
    p=30,
    h=21,
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
    Train an MLP for h-step-ahead volatility forecasting and add the resulting forecast
    series to the input df.
    Model trained only on observations whose target dates lies in the training sample.
    After training: forecasts generated for all valid target dates and stored in a new column.
    """
    # ste random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = df.copy()

    # build the feature matrix and target vector
    X, y = make_lagged_matrix_h(df, hv_col=hv_col, p=p, h=h, use_vdax=False, vdax_col="VDAX")

    split_ts = pd.Timestamp(split_date)

    # X indexed by origin days t, goal to be store at their target dates t+h
    origin_pos = df.index.get_indexer(X.index)      # integer positions of origin dates t
    target_pos = origin_pos + h                     # corresponding pos of target dates t+h

    # keep only pairs for wich t+h still lies inside the DataFrame
    valid = target_pos < len(df)
    X = X.iloc[valid]
    y = y.iloc[valid]

    # convert valid target positions back into actual index values
    target_idx = df.index[target_pos[valid]]    
    

    # reindex X and y so that both are now indexed by target date t+h
    X = X.set_axis(target_idx)  
    y = y.set_axis(target_idx)  

    # use the actual calander dates to define the train/test split
    target_dates = df.loc[target_idx, "Date"]
    train_mask = target_dates < split_ts
    test_mask = target_dates >= split_ts

    # standardize input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.loc[train_mask])
    X_test = scaler.transform(X.loc[test_mask])  # not used for training, but keeps flow consistent

    # training targets as a column vector
    y_train = y.loc[train_mask].values.reshape(-1, 1)

    # convert training data to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    # wrap tensors in a dataset and dataloader for mini-batch training
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize model, loss function and optimizer
    model = MLP(input_dim=X.shape[1], hidden_dims=hidden_dims)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # variables for early stopping
    best_loss = np.inf
    counter = 0
    best_state = None

    # TRAINING loop
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0

        # iterate over mini-batches
        for xb, yb in loader:
            optimizer.zero_grad()           # reset gradients
            pred = model(xb)                # forward pass
            loss = criterion(pred, yb)      # compute batch loss
            loss.backward()                 # backpropagation
            optimizer.step()                # parameter update
            epoch_loss += loss.item()

        # average loss over all mini-batches in this epoch
        epoch_loss /= len(loader)

        # save model weights whenever a new best training loss is reached
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        # stop if training loss has not improved for 'patience' epochs
        if counter >= patience:
            break
    
    # restore the besto model encountered during training
    model.load_state_dict(best_state)
    model.eval()

    # generate forecasts for the full valid sample (train+ test)
    X_all = scaler.transform(X)
    X_all_t = torch.tensor(X_all, dtype=torch.float32)

    with torch.no_grad():
        yhat = model(X_all_t).numpy().flatten()

    # store forecasts in new column
    colname = f"MLP_Forecast_h{h}"
    df[colname] = np.nan
    df.loc[target_idx, colname] = yhat

    # apply lower and upper bounds to avoid implausible forecast values
    hv = df[hv_col].to_numpy(dtype=float)
    vol_cap = np.nanquantile(hv, 0.999)
    vol_cap = float(vol_cap) if np.isfinite(vol_cap) else None
    df[colname] = clip_forecast_series(df[colname], vol_floor=vol_floor, vol_cap=vol_cap)

    return df

#############################################################################################################
## MAIN
#############################################################################################################

if __name__ == "__main__":

    split_date = pd.Timestamp("2018-01-01")

    df = load_and_prepare_data(
        path = "DAX Historical Data.csv",
        sep=";",
        decimal=",",
        date_format="%m.%d.%Y",
        drop_cols=["Open", "High", "Low", "Vol,", "Change %"]
    )


    # produce both horizons in one run
    for h in (21, 63):
        df = add_mlp_forecast_h(
            df=df,
            hv_col="HV_30d",
            p=30,
            h=h,
            split_date=split_date,
            hidden_dims=(32, 16),
            lr=1e-3
        )

    # keep only columns needed for export and plotting
    out_cols = ["Date", "HV_30d", "MLP_Forecast_h21", "MLP_Forecast_h63"]
    df_out = df[out_cols].dropna(subset=["Date", "HV_30d", "MLP_Forecast_h21", "MLP_Forecast_h63"])

    # label observations as training or test sample based on calendar date
    df_out["Sample"] = np.where(df_out["Date"] < split_date, "Train", "Test")

    # export of the series
    #df_out.to_csv("outputs/mlp_forecasts_h.csv", index=False, sep=";")
    #print("mlp_forecasts_h.csv got created.")

    # copy for plotting
    df =  df_out.copy()
    df["Date"] = pd.to_datetime(df["Date"])


    df_oos = df[df["Date"] >= split_date]
  

    plot_volatility_forecasts(
            df=df_oos,
            forecast_specs=[
                (f"MLP_Forecast_h21", "MLP"),
            ],
            asset_name="DAX",
            model_name="MLP",
            horizon=21,
            split_date=split_date,
            oos_only=True,
            save_path=f"plots/dax_mlp_h21.png",
        )
    
    plot_volatility_forecasts(
            df=df_oos,
            forecast_specs=[
                (f"MLP_Forecast_h63", "MLP"),
            ],
            asset_name="DAX",
            model_name="MLP",
            horizon=63,
            split_date=split_date,
            oos_only=True,
            save_path=f"plots/dax_mlp_h63.png",
        )

    
