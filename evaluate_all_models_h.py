import argparse
import numpy as np
import pandas as pd
from econometric_models_h import vol_floor

##################################################################
# script for evaluating all models in the k-step ahead approach
##################################################################

VAR_FLOOR = vol_floor**2

def sanity_report(df: pd.DataFrame, cols, name=""):
    """
    Print compact diagnostic summary for a list of columns
    """
    rep = []
    for c in cols:
        # marks missing columns so merge problems become visible
        if c not in df.columns:
            rep.append({"col": c, "status": "MISSING"})
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        rep.append({
            "col": c,
            "status": "OK",
            "n": int(x.shape[0]),
            "n_nan": int(x.isna().sum()),
            "n_inf": int(np.isinf(x.to_numpy(dtype=float)).sum()),
            "min": float(np.nanmin(x)) if np.isfinite(x).any() else np.nan,
            "p01": float(np.nanquantile(x, 0.01)) if np.isfinite(x).any() else np.nan,
            "p50": float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan,
            "p99": float(np.nanquantile(x, 0.99)) if np.isfinite(x).any() else np.nan,
            "max": float(np.nanmax(x)) if np.isfinite(x).any() else np.nan,
        })
    out = pd.DataFrame(rep)
    if name:
        print(f"\n--- Sanity report: {name} ---")
    print(out.to_string(index=False))
    return out

# -------------------------
# Metrics
# -------------------------
def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def qlike(y_true, y_pred, var_floor=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    var_true = y_true**2
    var_pred = y_pred**2

    if var_floor is None:
        # robuster Floor: z.B. 0.1% Quantil der REALISIERTEN Varianz
        vt = var_true[np.isfinite(var_true) & (var_true > 0)]
        var_floor = float(np.nanquantile(vt, 0.001)) if len(vt) else 1e-8

    var_pred = np.clip(var_pred, var_floor, None)

    out = np.log(var_pred) + var_true / var_pred
    out = out[np.isfinite(out)]
    return float(np.mean(out))

# -------------------------
# DM test (optional)
# -------------------------
def newey_west_variance(x: np.ndarray, lag: int) -> float:
    """
    Newey-West long-run variance estimate of a mean-zero series x.
    """
    T = len(x)
    if T < 2:
        return np.nan

    x = x - np.mean(x)
    gamma0 = np.dot(x, x) / T
    lrv = gamma0

    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = np.dot(x[k:], x[:-k]) / T
        lrv += 2.0 * w * gamma_k

    return float(lrv)

def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, nw_lag: int = 5):
    """
    Diebold-Mariano test for equal predictive accuracy:
    H0: E[loss_a - loss_b] = 0
    Returns (dm_stat, p_value_approx_normal).
    """
    from math import erf, sqrt

    d = loss_a - loss_b
    T = len(d)
    if T < 5:
        return (np.nan, np.nan)

    lrv = newey_west_variance(d, lag=min(nw_lag, T - 2))
    if not np.isfinite(lrv) or lrv <= 0:
        return (np.nan, np.nan)

    dm_stat = np.mean(d) / np.sqrt(lrv / T)

    def phi(z):
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    p = 2.0 * (1.0 - phi(abs(dm_stat)))
    return (float(dm_stat), float(p))

# -------------------------
# Input and output helpers
# -------------------------
def read_forecast_csv(path: str, date_col="Date") -> pd.DataFrame:
    """
    Read a forecast csv, parse the date column and sort by date
    """
    df = pd.read_csv(path, sep=";")
    if date_col not in df.columns:
        raise ValueError(f"{path}: missing '{date_col}' column.")
    df[date_col] = pd.to_datetime(df[date_col])
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
    Compute evaluation metrics for all available forecast columns in test set.
    """
    y_true_all = pd.to_numeric(df_test[hv_col], errors="coerce").to_numpy(dtype=float)

    # Horizon-specific robust variance floor from realized variance
    vt = y_true_all[np.isfinite(y_true_all)]
    vt = vt[vt > 0] ** 2
    var_floor = float(np.nanquantile(vt, 0.001)) if len(vt) else 1e-8

    rows = []
    for model, col in model_cols.items():
        if col not in df_test.columns:
            continue

        y_pred_all = pd.to_numeric(df_test[col], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
        yt = y_true_all[mask]
        yp = y_pred_all[mask]

        if len(yt) == 0:
            continue

        # do not clip predictions for MSE/ RMSE/ MAE
        rows.append({
            "Model": model,
            "N": int(len(yt)),
            "MSE": mse(yt, yp),
            "RMSE": rmse(yt, yp),
            "MAE": mae(yt, yp),
            # QLIKE: Floor nur hier, konsistent pro Horizon
            "QLIKE": qlike(yt, yp, var_floor=var_floor),
        })

    if not rows:
        return pd.DataFrame(columns=["Model", "N", "MSE", "RMSE", "MAE", "QLIKE"])

    return (
        pd.DataFrame(rows)
        .sort_values(["QLIKE", "RMSE"], ascending=True)
        .reset_index(drop=True)
    )

def parse_horizons(s: str):
    """
    Parse a comma-separted horizon string into a list of integers.
    """
    return [int(x.strip()) for x in s.split(",") if x.strip()]

#############################################################################################################
## MAIN
#############################################################################################################

def main():
    """
    Main evaluation pipline. 

    For each forecast horizon.
    1. load benchmark, MLP, LSTM files
    2. merge them on the date column
    3. built a consistent observed volatiltiy column
    4. run diagnostic checks
    5. compute loss functions
    6. opt DM test against chosen baeline
    7. save merged files and metric tables
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2018-01-01")
    #f"lstm_hybrid_forecasts_h{horizon}.csv"
    
    # Input files
    ap.add_argument("--bench", type=str, default="outputs/benchmark_forecasts_h.csv")
    ap.add_argument("--mlp", type=str, default="outputs/mlp_forecasts_h.csv")
    ap.add_argument("--lstm-template", type=str, default="outputs/lstm_hybrid_forecasts_h{h}.csv")
    
    # Horizon selection + output paths
    ap.add_argument("--horizons", type=str, default="21,63")
    ap.add_argument("--out-merged-template", type=str, default="outputs/all_forecasts_merged_h{h}.csv")
    ap.add_argument("--out-metrics-template", type=str, default="outputs/forecast_evaluation_metrics_h{h}.csv")
    ap.add_argument("--out-metrics-all", type=str, default="outputs/forecast_evaluation_metrics_all_horizons.csv")

    # Optional DM tests
    ap.add_argument("--dm", action="store_true", help="Run DM tests vs a baseline (per horizon).")
    ap.add_argument("--dm-baseline", type=str, default="RW")  # baseline MODEL NAME (not column)
    ap.add_argument("--dm-loss", type=str, default="se", choices=["se", "ae", "qlike"])
    ap.add_argument("--dm-lag", type=int, default=5)
    ap.add_argument("--out-dm-template", type=str, default="outputs/dm_tests_h{h}.csv")

    args = ap.parse_args()
    print("\n--- DEBUG ARGS ---")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("------------------\n")
    split_date = pd.Timestamp(args.split_date)
    horizons = parse_horizons(args.horizons)

    # Load shared files once
    df_bench = read_forecast_csv(args.bench)
    #df_split = read_forecast_csv(args.split)
    df_mlp = read_forecast_csv(args.mlp)

    all_metrics = []

    for h in horizons:
        lstm_path = args.lstm_template.format(h=h)
        df_lstm = read_forecast_csv(lstm_path)
        lstm_keep_cols = [
            "Date",
            f"LSTM_Forecast_h{h}",
            f"G-LSTM_Forecast_h{h}",
            f"E-LSTM_Forecast_h{h}",
        ]
        df_lstm = df_lstm[[c for c in lstm_keep_cols if c in df_lstm.columns]].copy()

        # merge: start from bench (it has HV_30d), attach split + mlp + lstm/hybrids
        #df = df_bench.merge(df_split, on="Date", how="left")
        df = df_bench.copy()
        df = df.merge(df_mlp, on="Date", how="left", suffixes=("", "_mlp"))
        df = df.merge(df_lstm, on="Date", how="left", suffixes=("", "_lstm"))
        #df = df.merge(df_lstm_split, on="Date", how="left", suffixes=("", "_lstmsplit"))

        # coalesce HV if needed (defensive)
        hv_sources = ["HV_30d", "HV_30d_mlp", "HV_30d_lstm"]
        if "HV_30d" not in df.columns:
            df = coalesce_hv(df, hv_sources)
        else:
            for alt in ["HV_30d_mlp", "HV_30d_lstm"]:
                if alt in df.columns:
                    df["HV_30d"] = df["HV_30d"].fillna(df[alt])

        # convert all non-date columns to numeric
        for c in df.columns:
            if c != "Date":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.sort_values("Date").reset_index(drop=True)

        # horizon specific columns to inspect after mergin
        check_cols_h = [
            "HV_30d",
            f"RW_Forecast_h{h}", f"EWMA_Forecast_h{h}",
            f"GARCH_Split_Forecast_h{h}", f"EGARCH_Split_Forecast_h{h}",
            f"MLP_Forecast_h{h}",
            f"LSTM_Forecast_h{h}",
            f"G-LSTM_Forecast_h{h}", f"E-LSTM_Forecast_h{h}", 
        ]
        sanity_report(df, check_cols_h, name=f"merged h={h}")

        # mapping from model names to actual forecast columns
        model_cols = {
            "RW": f"RW_Forecast_h{h}",
            "EWMA": f"EWMA_Forecast_h{h}",
            "GARCH (split)": f"GARCH_Split_Forecast_h{h}",
            "EGARCH (split)": f"EGARCH_Split_Forecast_h{h}",
            "MLP": f"MLP_Forecast_h{h}",
            "LSTM": f"LSTM_Forecast_h{h}",
            "G-LSTM": f"G-LSTM_Forecast_h{h}",
            "E-LSTM": f"E-LSTM_Forecast_h{h}",
        }

        # restric evaluation to out-of-sample period
        df_test = df[df["Date"] >= split_date].copy()

        # compute evaluation metrics for all available models
        metrics = evaluate_table(df_test, hv_col="HV_30d", model_cols=model_cols)
        metrics.insert(0, "Horizon", h)

        # save in merged file and metrics table
        out_merged = args.out_merged_template.format(h=h)
        out_metrics = args.out_metrics_template.format(h=h)
        df.to_csv(out_merged, index=False, sep=";")
        metrics.to_csv(out_metrics, index=False, sep=";")

        print(f"\n[h={h}] Merged forecasts written to: {out_merged} (rows={len(df)})")
        print(f"[h={h}] Metrics written to:         {out_metrics}")
        if len(metrics):
            print(metrics.drop(columns=["Horizon"]).to_string(index=False))
        else:
            print(f"[h={h}] No metrics computed (no overlapping non-NaN pairs in test set).")

        all_metrics.append(metrics)

        # Optional DM tests per horizon
        if args.dm and len(metrics):
            # baseline specified by MODEL NAME (e.g., RW)
            if args.dm_baseline not in model_cols:
                print(f"[h={h}] DM: baseline model '{args.dm_baseline}' not in model map. Skipping DM.")
                continue

            base_col = model_cols[args.dm_baseline]
            if base_col not in df_test.columns:
                print(f"[h={h}] DM: baseline column '{base_col}' not found. Skipping DM.")
                continue

            y = df_test["HV_30d"].values
            base = df_test[base_col].values

            def loss_series(y_true, y_pred, kind):
                """
                Build observation-wise lsos series used in the DM test.
                """
                if kind == "se":
                    return (y_pred - y_true) ** 2
                if kind == "ae":
                    return np.abs(y_pred - y_true)
                if kind == "qlike":
                    # horizon-specific floor derived from observed variance
                    vt = y_true[np.isfinite(y_true)]
                    vt = vt[vt > 0] ** 2
                    var_floor = float(np.nanquantile(vt, 0.001)) if len(vt) else 1e-8

                    var_pred = np.clip(y_pred**2, var_floor, None)
                    var_true = y_true**2
                    return np.log(var_pred) + var_true / var_pred

            dm_rows = []
            for model_name, col in model_cols.items():
                if model_name == args.dm_baseline:
                    continue
                if col not in df_test.columns:
                    continue
                pred = df_test[col].values
                mask = np.isfinite(y) & np.isfinite(pred) & np.isfinite(base)
                if mask.sum() < 10:
                    continue
                la = loss_series(y[mask], pred[mask], args.dm_loss)
                lb = loss_series(y[mask], base[mask], args.dm_loss)
                stat, p = dm_test(la, lb, nw_lag=args.dm_lag)
                dm_rows.append({
                    "Horizon": h,
                    "Model": model_name,
                    "Compared_to": args.dm_baseline,
                    "Loss": args.dm_loss,
                    "N": int(mask.sum()),
                    "DM_stat": stat,
                    "p_value": p
                })

            dm_df = pd.DataFrame(dm_rows).sort_values("p_value", ascending=True).reset_index(drop=True)
            out_dm = args.out_dm_template.format(h=h)
            dm_df.to_csv(out_dm, index=False, sep=";")
            print(f"[h={h}] DM tests written to:      {out_dm}")
            if len(dm_df):
                print(dm_df.to_string(index=False))

    # save one combined metrics file across horizons
    if all_metrics:
        metrics_all = pd.concat(all_metrics, ignore_index=True)
        metrics_all.to_csv(args.out_metrics_all, index=False, sep=";")
        print(f"\nCombined metrics written to: {args.out_metrics_all} (rows={len(metrics_all)})")



if __name__ == "__main__":
    main()



# Diebold Test starten: im terminal mit --dm flag
# 1. python evaluate_all_models_h.py --dmm
# 2. opt: baseline-modell festlegen:
#    -> python evaluate_all_models_h.py --dm --dm-baseline RW 
# python evaluate_all_models_h.py --dm --horizons 21 --dm-baseline "G-LSTM" --dm-loss qlike
#    -> (aussage: RW "bestes" model, vergleichen welches doch besser?)
#    -> nun weden alle modelle gegen das baseline-model getestet, hier : alle modelle vs RW
# 3. gespeichert wird in: outputs/dm_tests_h{h}.csv