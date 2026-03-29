import math
import numpy as np
import pandas as pd


##################################################################
# script for comparison of mlp_forecast_h.py and mlp_h.py
# mlp_forecast_h.py: simpler feature set
# mlp_h: richer feature set
##################################################################


# -----------------------------
# Configuration
# -----------------------------

split_date = pd.Timestamp("2018-01-01")
simple_path = "outputs/mlp_forecasts_h.csv"   # simpler MLP file
rich_paths = {
    21: "outputs/mlp_forecasts_h21.csv",      # richer MLP horizon 21
    63: "outputs/mlp_forecasts_h63.csv",      # richer MLP horizon 63
}

output_path = "outputs/dm_mlp_feature_sets.csv"
sep = ";"
eps = 1e-12

# if set to none, newey-west lag is choosen as h-1
fixed_nw_lag = None


# -----------------------------
# Loss functions per observation: 
# qlike, squared error calculations
# -----------------------------

def qlike_series(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true = np.clip(y_true, eps, None)
    y_pred = np.clip(y_pred, eps, None)

    sigma2_true = y_true ** 2
    sigma2_pred = y_pred ** 2

    return np.log(sigma2_pred) + sigma2_true / sigma2_pred


def squared_error_series(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return (y_true - y_pred) ** 2


def get_loss_series(y_true, y_pred, loss_name):
    """
    Dispatch to the requested loss function and return the resulting
    loss differential series
    """
    loss_name = loss_name.lower()

    if loss_name == "qlike":
        return qlike_series(y_true, y_pred, eps=eps)
    elif loss_name == "se":
        return squared_error_series(y_true, y_pred)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# -----------------------------
# DM test helpers
# -----------------------------

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def newey_west_long_run_variance(d, lag):
    """
    Compute Newey-West heteroskeadticity and autocrrelation consistent long-run variance
    estimator for the loss differential d_f. needed because multi-step ahead forecast loss
    differentials are typically serially correlated.
    """
    d = np.asarray(d, dtype=float)
    T = len(d)

    if T < 2:
        raise ValueError("Need at least 2 observations.")

    # restrict lag length to a feasible range
    lag = int(max(0, min(lag, T - 1)))
    d_centered = d - np.mean(d)

    # contemporaneous covariance
    gamma0 = np.dot(d_centered, d_centered) / T
    lrv = gamma0

    # add weighted autocovariances using bratlett weights
    for j in range(1, lag + 1):
        gamma_j = np.dot(d_centered[j:], d_centered[:-j]) / T
        weight = 1.0 - j / (lag + 1.0)
        lrv += 2.0 * weight * gamma_j

    # ensure var stays positive
    return max(lrv, 1e-18)


def diebold_mariano_test(loss_model, loss_benchmark, h=1, nw_lag=None, harvey_adjust=True):
    """
    DM test with MLP_simpe as benchmark vs MLP_rich
    """
    loss_model = np.asarray(loss_model, dtype=float)
    loss_benchmark = np.asarray(loss_benchmark, dtype=float)

    # only pbservations wehere both loss series are finite
    mask = np.isfinite(loss_model) & np.isfinite(loss_benchmark)
    loss_model = loss_model[mask]
    loss_benchmark = loss_benchmark[mask]

    if len(loss_model) != len(loss_benchmark):
        raise ValueError("Loss vectors must have same length.")

    d = loss_model - loss_benchmark
    T = len(d)

    if T < 5:
        raise ValueError("Too few observations for DM test.")

    # for h-step ahead forecast, the default newey-west lag is h-1
    if nw_lag is None:
        nw_lag = max(h - 1, 0)

    # estimate the long-run variance of the loss differential
    lrv = newey_west_long_run_variance(d, lag=nw_lag)

    # dm stat
    dm_stat = np.mean(d) / math.sqrt(lrv / T)

    # opt Harvey-Leybourne-Newbold adjustment
    if harvey_adjust and h > 1:
        factor_num = T + 1 - 2 * h + (h * (h - 1)) / T
        factor_den = T
        if factor_num > 0:
            dm_stat *= math.sqrt(factor_num / factor_den)

    p_value = 2.0 * (1.0 - normal_cdf(abs(dm_stat)))

    return dm_stat, p_value


# ---------------------------------
# Load forecast files
# ---------------------------------

def load_simple_file(path, h):
    """
    Load the forecast file from simpler MLP specification for given horizon h and 
    rename columns
    """
    df = pd.read_csv(path, sep=sep)
    df["Date"] = pd.to_datetime(df["Date"])

    pred_col = f"MLP_Forecast_h{h}"
    needed = ["Date", "HV_30d", pred_col]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in simpler MLP file for h={h}: {missing}")

    df = df[needed].copy()
    df = df.rename(columns={
        "HV_30d": "HV_30d_simple",
        pred_col: "MLP_simple"
    })
    return df


def load_rich_file(path, h):
    """
    Load the forecast file from richer MLP specification for given horizon h and 
    rename columns
    """
    df = pd.read_csv(path, sep=sep)
    df["Date"] = pd.to_datetime(df["Date"])

    pred_col = f"MLP_Forecast_h{h}"
    needed = ["Date", "HV_30d", pred_col]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in richer MLP file for h={h}: {missing}")

    df = df[needed].copy()
    df = df.rename(columns={
        "HV_30d": "HV_30d_rich",
        pred_col: "MLP_rich"
    })
    return df


# -----------------------------
# Main comparison
# -----------------------------

def run_dm_for_feature_sets():
    """
    Compare richer and simpler MLP forecast specification using DM test for each
    forecast horizon.

    For each horizon:
    1. load corresponding files
    2. merge them by date
    3. restrict the sample to out of sample period
    4. compute loss differentials under QLIKE and squared error
    5. run DM test
    """
    rows = []

    for h, rich_path in rich_paths.items():
        simple_df = load_simple_file(simple_path, h=h)
        rich_df = load_rich_file(rich_path, h=h)

        # merge both forecast files on their common dtes
        df = simple_df.merge(rich_df, on="Date", how="inner")

        # keep only out-of-sample observations
        df = df[df["Date"] >= split_date].copy()

        # remove any rows with missing observed values or forecasts
        df = df.dropna(subset=[
            "HV_30d_simple",
            "HV_30d_rich",
            "MLP_simple",
            "MLP_rich"
        ])

        if df.empty:
            raise ValueError(f"No overlapping out-of-sample observations for h={h}.")

        # Use the simple file's observed HV as reference
        y_true = df["HV_30d_simple"].to_numpy()
        y_pred_simple = df["MLP_simple"].to_numpy()
        y_pred_rich = df["MLP_rich"].to_numpy()

        # run DM test under both loss functions
        for loss_name in ["qlike", "se"]:
            loss_benchmark = get_loss_series(y_true, y_pred_simple, loss_name)
            loss_model = get_loss_series(y_true, y_pred_rich, loss_name)

            dm_stat, p_value = diebold_mariano_test(
                loss_model=loss_model,
                loss_benchmark=loss_benchmark,
                h=h,
                nw_lag=fixed_nw_lag,
                harvey_adjust=True
            )

            rows.append({
                "Horizon": h,
                "Model": "MLP_rich",
                "Compared to": "MLP_simple",
                "Loss": loss_name,
                "DM stat": dm_stat,
                "p-value": p_value
            })

    # collect test results in one summary table
    result = pd.DataFrame(rows)
    result = result[["Horizon", "Model", "Compared to", "Loss", "DM stat", "p-value"]]
    result = result.sort_values(["Horizon", "Loss"]).reset_index(drop=True)

    return result


if __name__ == "__main__":
    dm_df = run_dm_for_feature_sets()

    print("\nDM test results: MLP_rich vs MLP_simple")
    print(dm_df)

    dm_df.to_csv(output_path, index=False, sep=sep)
    print(f"\nSaved DM results to: {output_path}")