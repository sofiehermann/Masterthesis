import argparse
import numpy as np
import pandas as pd
from math import erf, sqrt

EPS = 1e-12

##################################################################
# This script evaluates all models with the loss function and procedure
# introduced in section 4.9
##################################################################

# -----------------------------------------------------------------
# Forecast evaluation metrics
# -----------------------------------------------------------------
def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def qlike(y_true, y_pred, var_floor=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    var_true = y_true**2
    var_pred = y_pred**2

    var_pred = np.clip(var_pred, var_floor, None)

    out = np.log(var_pred) + var_true / var_pred
    out = out[np.isfinite(out)]
    return float(np.mean(out))

# -----------------------------------------------------------------
# Diebold-Mariano test support
# -----------------------------------------------------------------
def newey_west_variance(x: np.ndarray, lag: int) -> float:
    """
    Newey-West long-run variance estimate of a mean-zero series x.
    Needed because d_t=loss_a,t-loss_b,t can be autocorrelated over time.
    Newey-West adjusts for autocorrelation up to a choosen lag.

    Returns: estimated long-run variance
    """
    T = len(x)

    # if series too short -> variance estiamte not meaningful
    if T < 2:
        return np.nan

    # center the series around zero
    x = x - np.mean(x)
    gamma0 = np.dot(x, x) / T
    lrv = gamma0

    # add weighted higher-lag autocovariances
    # weights decrease linarly with lag
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = np.dot(x[k:], x[:-k]) / T
        lrv += 2.0 * w * gamma_k

    return float(lrv)

def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, nw_lag: int = 5):
    """
    Diebold-Mariano test for equal predictive accuracy:
    H0: E[loss_a - loss_b] = 0

    Returns: (dm_stat, p_value_approx_normal) = (test statistic, p-value)
    """
    
    d = loss_a - loss_b
    T = len(d)
    if T < 5:
        return (np.nan, np.nan)

    lrv = newey_west_variance(d, lag=min(nw_lag, T - 2))

    # if var estimate is invalid or non-postive, test stat cannot be computed
    if not np.isfinite(lrv) or lrv <= 0:
        return (np.nan, np.nan)

    # DM test statistic
    dm_stat = np.mean(d) / np.sqrt(lrv / T)

    # standard normal cdf using erf
    def phi(z):
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    p = 2.0 * (1.0 - phi(abs(dm_stat))) # two sided p-value

    print("DM stat:", dm_stat)
    print("p-value:", p)
    print("------")

    return (float(dm_stat), float(p))


# ---------------------------------------------------------------------------
# Input / output helper functions
# ---------------------------------------------------------------------------
def read_forecast_csv(path: str, date_col="Date") -> pd.DataFrame:
    """
    Read a forecast CSV file using semicolon seperator, pars date column, sort by date
    and return the result.

    Returns: cleaned datafram with sorted dates
    """
    df = pd.read_csv(path, sep=";")

    # check if date column exists
    if date_col not in df.columns:
        raise ValueError(f"{path}: missing '{date_col}' column.")
    
    # convert date column to pandas datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # sort by time so later merges and evaluations are chronologically ordered
    df = df.sort_values(date_col).reset_index(drop=True)
    return df

def coalesce_hv(df: pd.DataFrame, hv_candidates):
    """
    Ensure df contains a single 'HV_30d' column by coalescing candidates if needed.
    """
    if "HV_30d" in df.columns:
        return df
    for c in hv_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "HV_30d"})
            return df
    raise ValueError("Could not find realized volatility column HV_30d in merged data.")

def evaluate_table(df_test: pd.DataFrame, hv_col: str, model_cols: dict) -> pd.DataFrame:
    """
    Compute evaluation metrics for all forecast models on test set.

    Returns: table with one row per model and columns for loss function
    """
    # observed vola series
    y_true = df_test[hv_col].values
    rows = []

    # loop over all models defined in model_cols
    for model, col in model_cols.items():
        if col not in df_test.columns:
            continue

        y_pred = df_test[col].values
        
        # pairsie removal of missing values 
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[mask]
        yp = y_pred[mask]

        # if no valid observations remain, skip this model
        if len(yt) == 0:
            continue

        # compute all metrics and store them as one row
        rows.append({
            "Model": model,
            "N": int(len(yt)),
            "MSE": mse(yt, yp),
            "RMSE": rmse(yt, yp),
            "MAE": mae(yt, yp),
            "QLIKE": qlike(yt, yp),
        })
    
    # create result and sort it: first by QLIKE, then by RMSE both ascending
    out = pd.DataFrame(rows).sort_values(["QLIKE", "RMSE"], ascending=True).reset_index(drop=True)
    return out


#############################################################################################################
## MAIN
#############################################################################################################

def main():

    """
    Main workflow of the script. It does the following:
    1. read forecast files from different models.
    2. merge them into one large dataframe by date
    3. clean and align the observed volatility column
    4. restrict evaluation to out-of-sample period
    5. compute forecast evaluation metrics
    6. opt run DM test against a chosen baseline model
    7. save output to csv.
    """
    # ---------------------------------
    # Parse command-line arguments
    # ---------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2018-01-01")

    # input files
    ap.add_argument("--bench", type=str, default="outputs/benchmark_forecasts.csv")
    ap.add_argument("--mlp", type=str, default="outputs/mlp_forecasts.csv")
    ap.add_argument("--lstmsplit", type=str, default="outputs/lstm_and_splithybrid_forecasts.csv")

    # output files
    ap.add_argument("--out-metrics", type=str, default="outputs/forecast_evaluation_metrics.csv")
    ap.add_argument("--out-merged", type=str, default="outputs/all_forecasts_merged.csv")

    # optional DM test stettings
    ap.add_argument("--dm", action="store_true", help="Run Diebold-Mariano tests vs baseline.")
    ap.add_argument("--dm-baseline", type=str, default="RW_Forecast")
    ap.add_argument("--dm-loss", type=str, default="se", choices=["se", "ae", "qlike"])
    ap.add_argument("--dm-lag", type=int, default=5)

    args = ap.parse_args()

    # convert split date to pandas timestamp for later comparison
    split_date = pd.Timestamp(args.split_date)

    # ---------------------------------
    # Load forecast files & Merge files by date
    # ---------------------------------
    
    # read forecast result files
    df_bench = read_forecast_csv(args.bench)
    df_mlp = read_forecast_csv(args.mlp)
    df_lstmsplit = read_forecast_csv(args.lstmsplit)

    # Merge on Date (outer keeps maximum coverage; evaluation will drop NaNs per model)
    df = df_bench.merge(df_mlp, on="Date", how="outer", suffixes=("", "_mlp"))
    df = df.merge(df_lstmsplit, on="Date", how="outer", suffixes=("", "_lstmsplit"))

    # If multiple HV columns exist (HV_30d, HV_30d_mlp), coalesce them to one
    hv_sources = ["HV_30d", "HV_30d_mlp", "HV_30d_lstmsplit"]
    if "HV_30d" not in df.columns:
        df = coalesce_hv(df, hv_sources)
    else:
        # Fill missing HV_30d using other sources if present
        for alt in ["HV_30d_mlp", "HV_30d_lstmsplit"]:
            if alt in df.columns:
                df["HV_30d"] = df["HV_30d"].fillna(df[alt])

    # convert non-date columns to numeric
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort once more after merging and cleaning
    df = df.sort_values("Date").reset_index(drop=True)

    # Define which forecast columns to evaluate (only those present will be used)
    model_cols = {
        # Benchmarks + (split/rolling if present)
        "RW": "RW_Forecast",
        "EWMA": "EWMA_Forecast",
        "GARCH (split)": "GARCH_Split_Forecast",
        "EGARCH (split)": "EGARCH_Split_Forecast",
        # MLP
        "MLP": "MLP_Forecast",
        # LSTM + Hybrids
        "LSTM": "LSTM_Forecast",
        "G-LSTM": "G-LSTM_Forecast",
        "E-LSTM": "E-LSTM_Forecast",
        "GE-LSTM": "GE-LSTM_Forecast",
    }

    # Test split, because evaluation only on test period
    df_test = df[df["Date"] >= split_date].copy()

    # Evaluate
    metrics = evaluate_table(df_test, hv_col="HV_30d", model_cols=model_cols)

    # Save merged + metrics
    df.to_csv(args.out_merged, index=False, sep=";")
    metrics.to_csv(args.out_metrics, index=False, sep=";")

    print("Columns in merged df:")
    print(sorted(df.columns))

    print("\nPresence check:")
    for c in ["GARCH_Split_Forecast","EGARCH_Split_Forecast"]:
        print(c, c in df.columns)

    print(f"Merged forecasts written to: {args.out_merged} (rows={len(df)})")
    print(f"Metrics written to:         {args.out_metrics}")
    print()
    print(metrics.to_string(index=False))

    # DM tests (optional)
    if args.dm:
        base_col = args.dm_baseline

        # if base_col not present, DM test cannot be run
        if base_col not in df_test.columns:
            print(f"\nDM: baseline column '{base_col}' not found in test data. Skipping DM tests.")
            return

        # actual and baseline forecast series
        y = df_test["HV_30d"].values
        base = df_test[base_col].values

        # identify observations where both actual and baseline are valid
        base_mask = np.isfinite(y) & np.isfinite(base)

        def loss_series(y_true, y_pred, kind):
            """
            Compute observation-by-observation loss values. This is different to them above,
            here we run the full time series of losses,
            because the DM test needs on loss value per observation
            """
            if kind == "se":
                return (y_pred - y_true) ** 2
            if kind == "ae":
                return np.abs(y_pred - y_true)
            if kind == "qlike":
                var_floor = 1e-8
                var_pred = np.clip(y_pred**2, var_floor, None)
                var_true = y_true**2
                return np.log(var_pred) + var_true / var_pred
            raise ValueError(kind)

        # baseline loss series
        base_loss = loss_series(y[base_mask], base[base_mask], args.dm_loss)

        dm_rows = []

        # compare each model against the baseline
        for model, col in model_cols.items():

            # skip if baseline model itself
            if col == base_col:
                continue

            # skip is forecast column is missing
            if col not in df_test.columns:
                continue

            pred = df_test[col].values

            # keep only dates where actual, candidate forecast and baseline forecast all exist
            mask = np.isfinite(y) & np.isfinite(pred) & np.isfinite(base)

            # require a minimum number of usable observations
            if mask.sum() < 10:
                continue

            # loss series for candidate model and baseline
            la = loss_series(y[mask], pred[mask], args.dm_loss)
            lb = loss_series(y[mask], base[mask], args.dm_loss)

            # run DM test
            stat, p = dm_test(la, lb, nw_lag=args.dm_lag)

            # store results
            dm_rows.append({
                "Model": model,
                "Compared_to": base_col,
                "Loss": args.dm_loss,
                "N": int(mask.sum()),
                "DM_stat": stat,
                "p_value": p
            })

        # create result dataframe sorted by p-value
        dm_df = pd.DataFrame(dm_rows).sort_values("p_value", ascending=True).reset_index(drop=True)

        # save dm results
        dm_out = "dm_tests.csv"
        dm_df.to_csv(dm_out, index=False, sep=";")
        print(f"\nDM tests written to: {dm_out}")
        if len(dm_df):
            print(dm_df.to_string(index=False))

if __name__ == "__main__":
    main()


# Diebold Test starten: im terminal mit --dm flag
# 1. python evaluate_all_models_h.py --dm
# 2. opt: baseline-modell festlegen:
#    -> python evaluate_all_models.py --dm --dm-loss qlike --dm-baseline RW_Forecast
#    -> (aussage: RW "bestes" model, vergleichen welches doch besser?)
#    -> nun weden alle modelle gegen das baseline-model getestet, hier : alle modelle vs RW
# 3. gespeichert wird in: outputs/dm_tests_h{h}.csv