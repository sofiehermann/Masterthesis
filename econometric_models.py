import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from tqdm import tqdm
import matplotlib.ticker as mtick
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from plot_forecasts import plot_volatility_forecasts, plot_log_returns, acf_plot, plot_distribution

VOL_FLOOR = 1e-6

##################################################################
# Sript for all econometric models for the one-step-ahead forecast
##################################################################


# -----------------------------------------------------------------
# Preparation of data 
# -----------------------------------------------------------------

def load_and_prepare_data(
        path,
        sep,
        decimal,
        date_format,
        drop_cols,
        price_col="Price",
        date_col="Date",
        hv_window=30
):
    """
    Loads data, calculates LogReturns and HV_30d.
    Returns dataframe with columns[Date, Price, LogReturn, HV_30d].
    """
    df = pd.read_csv(
        path,
        sep=sep,
        dtype={date_col: str},
        decimal=decimal,
    )

    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    # Dateformat 
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)

    # Sort: oldest to lates date
    df = df.sort_values(date_col).reset_index(drop=True)

    # LogReturn r_t = log(P_t / P_{t-1})
    df["LogReturn"] = np.log(df[price_col] / df[price_col].shift(1))

    # Historical vola (30 day rolling std)
    df["HV_30d"] = df["LogReturn"].rolling(window=hv_window).std(ddof=0)

    # Drop first NaNs
    df = df.dropna(subset=["LogReturn","HV_30d"])

    return df


def give_descriptive_stats(series):
    """
    Basic descriptive stats used in Section 2.3 (returns & volatility).
    """
    s = pd.Series(series).dropna()
    return pd.Series({
        "n": len(s),
        "mean": s.mean(),
        "std": s.std(ddof=0),
        "min": s.min(),
        "max": s.max(),
        "skew": s.skew(),
        "kurtosis_excess": s.kurtosis(),  # pandas gives excess kurtosis
    })


# -----------------------------------------------------------------
# Volatility Forecasting models
# First RW and EWMA then GARCH and EGARCH
# -----------------------------------------------------------------

def add_baseline_forecast(df, hv_col="HV_30d", lambda_=0.94):
    """
    Adds Random Walk and EWMA forecasts to the dataframe.
    Needs the columns 'LogReturn' and HV_30d.
    """
    df = df.copy()

    # RW Forecast: Vola yesterday is prediction for tomorrow
    df["RW_Forecast"] = df[hv_col].shift(1)

    # EWMA
    alpha = 1- lambda_
    r2 = df["LogReturn"] ** 2
    ewma_var = r2.ewm(alpha=alpha, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)
    df["EWMA_Forecast"] = ewma_vol.shift(1)

    return df


# -----------------------------------------------------------------
# GARCH and EGARCH Models
# Full sample, split sampel and rolling window
# -----------------------------------------------------------------

def add_garch_forecast(df, scale=100.0, return_model=False):
    """
    Adds the full-sample-GARCH(1,1) forecast series to the dataframe. 
    """

    df = df.copy()

    # scale returns
    returns = (df["LogReturn"] * scale).astype(float)

    garch = arch_model(
        returns,
        vol="GARCH",
        p=1,
        q=1,
        mean="Constant",
        dist="normal",
        rescale=False
    )

    garch_res = garch.fit(disp="off")

    VAR_FLOOR_SCALED = (VOL_FLOOR * scale) ** 2

    # reindex=True + start=0 -> forecast series over whole sample
    fcast = garch_res.forecast(horizon=1, start=0, reindex=True)

    # 1-step-ahead variance (scaled)
    var_1step_scaled = fcast.variance["h.1"].copy()
    # clip from below
    var_1step_scaled = var_1step_scaled.clip(lower=VAR_FLOOR_SCALED)

    # 1-step-ahead volatility in levels
    vol_1step = np.sqrt(var_1step_scaled) / scale

    df["GARCH_Forecast"] = (
        vol_1step.shift(1)
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=VOL_FLOOR)
    )

    if return_model:
        return df, garch_res
    return df

def add_egarch_forecast(df, scale=100.0, return_model=False):
    """
    Adds the full-sample-EGARCH(1,1) forecast series to the dataframe. 
    """
    df = df.copy()
    # again: LogReturn gets scaled
    returns = (df["LogReturn"] * scale).astype(float)

    egarch = arch_model(
        returns,
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        mean="Constant",
        dist="normal",
        rescale=False
    )

    egarch_res = egarch.fit(disp="off")

    VAR_FLOOR_SCALED = (VOL_FLOOR * scale) ** 2

    # full sample
    fcast = egarch_res.forecast(horizon=1, start=0, reindex=True)
    var_1step_scaled = fcast.variance["h.1"].copy()
    var_1step_scaled = var_1step_scaled.clip(lower=VAR_FLOOR_SCALED)

    # back into levels and volatility
    vol_1step = np.sqrt(var_1step_scaled) / scale

    df["EGARCH_Forecast"] = (
        vol_1step.shift(1)
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=VOL_FLOOR)
    )

    if return_model:
        return df, egarch_res
    return df

def add_split_garch_forecast(
        df,
        split_date,
        scale=100.0,
        forecast_col="GARCH_Split_Forecast"
    ):
    """
    Split-GARCH(1,1).
    Estimates GARCH(1,1) only on dates bevor split_date and
    creates from split_date on real one-step-ahead forecasts on the basis
    of the estimated parameters.
    - split_date: z.B. "2018-01-01"
    - forecast_col: column name for forecast (Standard: "GARCH_Split_Forecast")
    """

    df = df.copy()

    # Make sure that date is a datetime-column
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    # sort by data
    df = df.sort_values("Date").reset_index(drop=True)

    split_date = pd.Timestamp(split_date)

    # Train/Test
    # Boolean-Serie, True for all values in _mask
    train_mask = df["Date"] < split_date
    test_mask  = df["Date"] >= split_date

    # scaled LogReturns
    returns_scaled = df["LogReturn"] * scale

    # Filters only LogReturns out of train_mask
    returns_train = returns_scaled[train_mask]

    # estimate GARCH(1,1) on train
    garch = arch_model(
        returns_train,
        vol="GARCH",
        p=1,
        q=1,
        mean="Constant",
        dist="normal"
    )

    # estimates parameters omega, alpha, beta, mü with Maximum Likehihood
    res = garch.fit(disp="off")

    # get estimated parameters
    params = res.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta  = params["beta[1]"]
    mu    = params["mu"]   # Konstanten-Mean

    # last train-date t0: day before split -> starting point for the forecast
    last_train_idx = returns_train.index[-1]

    # conditional volaitlity and eps at the last train-date
    sigma_t = res.conditional_volatility.loc[last_train_idx]  
    sigma2_t = sigma_t ** 2
    eps_t = returns_scaled.loc[last_train_idx] - mu

    garch_forecast = np.full(len(df), np.nan)

    # from day after last_train_idx through the time (future)
    # at each day garch-recursion
    # forecast for date t+1 is sigma_{t+1}/scale
    full_index = df.index

    # position of last_train_idx in dataframe
    pos_last_train = full_index.get_loc(last_train_idx)
    VAR_FLOOR_SCALED = (VOL_FLOOR * scale) ** 2

    # loop from day after last train_day until the end
    for pos in range(pos_last_train + 1, len(full_index)):
        idx = full_index[pos]      # uptodate date is t+1 in df

        sigma2_next = omega + alpha * (eps_t ** 2) + beta * sigma2_t
        if not np.isfinite(sigma2_next):
            sigma2_next = VAR_FLOOR_SCALED
        sigma2_next = max(float(sigma2_next), VAR_FLOOR_SCALED)

        sigma_next = np.sqrt(sigma2_next)
        garch_forecast[pos] = max(float(sigma_next / scale), VOL_FLOOR)

        eps_t   = returns_scaled.loc[idx] - mu
        sigma2_t = sigma2_next

    # forecast colum into dataframe
    df[forecast_col] = garch_forecast

    return df

def add_split_egarch_forecast(
    df,
    split_date,
    scale=100.0,
    forecast_col="EGARCH_Split_Forecast",
    cap_quantile=0.999
):
    
    """
    Split EGARHC(1,1).
    Estimates a EGARCH(1,1) only on the data before the split_date.
    Creates from split_date on real one-step-ahead forecasts on the basis
    of the estimated parameters.
    - split_date: z.B. "2018-01-01"
    - forecast_col: column name for forecast (Standard: "EGARCH_Split_Forecast")
    """
    df = df.copy()

    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    split_date = pd.Timestamp(split_date)
    train_mask = df["Date"] < split_date

    returns_scaled = (df["LogReturn"] * scale).astype(float)
    returns_train = returns_scaled[train_mask]

    if len(returns_train) < 50:
        raise ValueError("Train sample too small for EGARCH estimation.")

    var_floor_scaled = (VOL_FLOOR * scale) ** 2

    # --- cap aus HV (oder fallback Returns) ---
    hv = df["HV_30d"].to_numpy(dtype=float)

    # upper boud for scaled variance:
    # 1. historical vola caled by scale and **2 to get variance
    # 2. cap_quantil as upper bound
    var_cap_scaled = np.nanquantile((hv * scale) ** 2, cap_quantile)
    # if calculated upper bound is not valid -> repleacement 
    if (not np.isfinite(var_cap_scaled)) or (var_cap_scaled <= var_floor_scaled):
        var_cap_scaled = np.nanquantile((returns_train - np.nanmean(returns_train)) ** 2, cap_quantile)

    # lower and upper variance bound into log-levels
    # because later the log variance gets used
    log_var_floor = float(np.log(var_floor_scaled))
    log_var_cap   = float(np.log(var_cap_scaled))

    # --- EGARCH fit auf Train ---
    egarch = arch_model(
        returns_train,
        vol="EGARCH",
        p=1, o=1, q=1,
        mean="Constant",
        dist="normal",
        rescale=False
    )
    res = egarch.fit(disp="off")

    p = res.params
    omega = float(p["omega"])
    alpha = float(p["alpha[1]"])
    gamma = float(p["gamma[1]"])
    beta  = float(p["beta[1]"])
    mu    = float(p["mu"])

    E_abs_z = float(np.sqrt(2.0 / np.pi))

    # starting point again last train day
    last_train_idx = returns_train.index[-1]
    # again sigma and epsilon on last train day and with an upper and lower bound
    sigma_t = float(res.conditional_volatility.loc[last_train_idx])  
    sigma2_t = float(np.clip(sigma_t * sigma_t, var_floor_scaled, var_cap_scaled))
    log_sigma2_t = float(np.clip(np.log(sigma2_t), log_var_floor, log_var_cap))
    eps_t = float(returns_scaled.loc[last_train_idx] - mu)

    # output values
    out = np.full(len(df), np.nan, dtype=float)

    pos_last_train = int(last_train_idx)  
    for pos in range(pos_last_train + 1, len(df)):
        sigma_t = float(np.sqrt(np.exp(log_sigma2_t)))
        sigma_t = max(sigma_t, float(np.sqrt(var_floor_scaled)))

        # standardized shock
        z_t = eps_t / sigma_t
        if not np.isfinite(z_t):
            z_t = 0.0

        # log-Varianz update (EGARCH recursion)
        log_sigma2_next = (
            omega
            + alpha * (abs(z_t) - E_abs_z)
            + gamma * z_t
            + beta * log_sigma2_t
        )

        # clip in log-space
        log_sigma2_next = float(np.clip(log_sigma2_next, log_var_floor, log_var_cap))

        # 1-step-ahead forecast 
        vol_next = float(np.sqrt(np.exp(log_sigma2_next)) / scale)
        out[pos] = max(vol_next, VOL_FLOOR)
        log_sigma2_t = log_sigma2_next
        eps_t = float(returns_scaled.iloc[pos] - mu)

    df[forecast_col] = out
    return df

def add_rolling_garch_forecast(
        df,
        window=252,
        scale=100.0, 
        forecast_col="Roll_GARCH_Forecast",
        mean: str = "Zero",
        dist: str = "normal"
    ):
    """
    Rolling GARCH(1,1).
    Estimates a Rolling GARCH(1,1) on the LogReturn and adds "Roll_GARCH_Forecast" to dataframe.
    For each window of length "Window" it fits a GARCH(1,1) and then produces a 
    1- step ahead forecast
    """
    df = df.copy()

    if "LogReturn" not in df.columns:
        raise ValueError("DataFrame must contain a 'LogReturn' column")
    
    if window is None or window <=10:
        raise ValueError("'window' must be an integer > 10")
    
    returns_scaled = (df["LogReturn"] * scale).astype(float)
    # scale returns as in add_garch_forecast
    n = len(returns_scaled)

    # Prepare a result array -> NaN when there is no foracast possibe
    roll_forecast = np.full(n, np.nan, dtype=float)

    start_i = window - 1
    end_i = n - 2

    for i in tqdm(range(start_i, end_i + 1), desc="Rolling GARCH"):
        # Select data from window: r_{t-W+1}, ..., r_t
        # First window: i-W+1 -> index 0,...,W-1
        # forecast gets written into i+1
        window_returns = returns_scaled.iloc[i - window + 1: i + 1]

        if not np.isfinite(window_returns.values).all():
            continue
        
        try:
        # Define GARCH(1,1) in this window
            garch = arch_model(
                    window_returns,
                    vol="GARCH",
                    p=1,
                    q=1,
                    mean=mean,
                    dist=dist,
                    rescale=False
            )

            # Fit GARCH model
            res = garch.fit(disp="off", show_warning=False)

            # 1-step-ahead forecast for variance of the *scaled* returns
            # reindex=False -> output length matches the window
            fcast = res.forecast(horizon=1, reindex=False)

            # fcast.variance is an array-like of shape (window_length, horizon)
            # takes last row (time t) and column 0 (1 step ahead horizon)
            var_next_scaled = float(fcast.variance.values[-1,0])

            if not np.isfinite(var_next_scaled) or var_next_scaled < 0:
                continue

            # Convert to volatility in the original scale
            vol_next = max(np.sqrt(var_next_scaled) / scale, VOL_FLOOR)
            # Store forecast at position i+1  (this is r_{t+1})
            roll_forecast[i + 1] = vol_next

        except Exception:
            continue

    # Add the new column to the DataFrame
    df[forecast_col] = roll_forecast

    return df


def add_rolling_egarch(
    df,
    window=252,
    scale=100.0,
    forecast_col="Roll_EGARCH_Forecast",
    mean: str = "Zero",
    dist: str = "normal",
    cap_quantile: float = 0.999,
    smooth_alpha: float = 0.0,   # 0.0 = kein Smoothing; z.B. 0.2 = leicht glätten
    force_fill: bool = True      # True: nach erstem validen Wert alle Lücken füllen
):
    
    """
    Rolling EGARCH(1,1).
    Estimates a Rolling EGARCH(1,1) on the LogReturn and adds "Roll_EGARCH_Forecast" to dataframe.
    For each window of length "Window" it fits a EGARCH(1,1) and then produces a 
    1- step ahead forecast
    """
      
    df = df.copy()

    if "LogReturn" not in df.columns:
        raise ValueError("Dataframe needs the column 'LogReturn'")
    if window is None or window <= 10:
        raise ValueError("'window' must be an integer > 10")

    r = (df["LogReturn"] * scale).astype(float)
    n = len(r)
    out = np.full(n, np.nan, dtype=float)

    var_floor_scaled = (VOL_FLOOR * scale) ** 2

    if "HV_30d" in df.columns:
        hv = df["HV_30d"].to_numpy(dtype=float)
        var_cap_scaled = np.nanquantile((hv * scale) ** 2, cap_quantile)
    else:
        var_cap_scaled = np.nanquantile((r - np.nanmean(r)) ** 2, cap_quantile)

    if (not np.isfinite(var_cap_scaled)) or (var_cap_scaled <= var_floor_scaled):
        var_cap_scaled = np.nanquantile((r - np.nanmean(r)) ** 2, 0.999)

    log_var_floor = float(np.log(var_floor_scaled))
    log_var_cap   = float(np.log(var_cap_scaled))

    def carry_forward(j: int):
        # take last valid forecast if existing
        if j > 0 and np.isfinite(out[j - 1]):
            out[j] = float(out[j - 1])

    start_i = window - 1
    end_i = n - 2  # write it into i+1

    for i in tqdm(range(start_i, end_i + 1), desc="Rolling EGARCH"):
        j = i + 1

        window_r = r.iloc[i - window + 1: i + 1]
        if not np.isfinite(window_r.values).all():
            carry_forward(j)
            continue

        try:
            model = arch_model(
                window_r,
                vol="EGARCH",
                p=1, o=1, q=1,
                mean=mean,
                dist=dist,
                rescale=False
            )
            res = model.fit(disp="off", show_warning=False)

            # check convergence 
            if getattr(res, "convergence_flag", 0) != 0:
                carry_forward(j)
                continue

            # extreme persistent fits tend to unstable 1-step forecasts
            params = getattr(res, "params", None)
            if params is not None and "beta[1]" in params and float(params["beta[1]"]) > 0.9995:
                carry_forward(j)
                continue

            fcast = res.forecast(horizon=1, reindex=False)
            var_next_scaled = float(fcast.variance.values[-1, 0])

            if (not np.isfinite(var_next_scaled)) or (var_next_scaled <= 0):
                carry_forward(j)
                continue

            # EGARCH is based on log-variance, so clips in log-space
            log_var_next = float(np.log(var_next_scaled))
            log_var_next = float(np.clip(log_var_next, log_var_floor, log_var_cap))
            var_next_scaled = float(np.exp(log_var_next))

            vol_next = float(np.sqrt(var_next_scaled) / scale)
            vol_next = max(vol_next, VOL_FLOOR)

            # optionales Smoothing
            if smooth_alpha and smooth_alpha > 0 and j > 0 and np.isfinite(out[j - 1]):
                vol_next = float(smooth_alpha * vol_next + (1.0 - smooth_alpha) * out[j - 1])

            out[j] = vol_next

        except Exception:
            carry_forward(j)
            continue

    s = pd.Series(out, index=df.index)

    if force_fill:
        # when first forecasts exisits, fill gap
        s = s.ffill()

    # final clip
    s = s.replace([np.inf, -np.inf], np.nan).clip(lower=VOL_FLOOR)

    df[forecast_col] = s
    return df


#############################################################################################################
## MAIN
#############################################################################################################

if __name__ == "__main__":

    df = load_and_prepare_data(
        path = "euro_stoxx_50.csv",
        sep=";",
        decimal=",",
        date_format="%m.%d.%Y",
        drop_cols=["Open", "High", "Vol,", "Change %"]
    )

    # call all forecasts
    df = add_baseline_forecast(df=df)
    df = add_garch_forecast(df=df)
    df = add_egarch_forecast(df=df)
    df = add_split_garch_forecast(df, split_date="2018-01-01")
    df = add_split_egarch_forecast(df=df, split_date="2018-01-01")
    #df = add_rolling_garch_forecast(df = df, window= 252,scale=100.0,forecast_col= "Roll_GARCH_Forecast", mean = "Zero",
     #   dist ="normal",cap_quantile=0.999, force_fill=True)
    
    #df = add_rolling_egarch(df=df, window=252, scale=100.0, forecast_col="Roll_EGARCH_Forecast", dist="normal",
      #  smooth_alpha=0.2)

    #all_cols = [
     #   "Date",
     #   "HV_30d",
     #   "RW_Forecast",
     #   "EWMA_Forecast",
     #   "GARCH_Split_Forecast",
     #   "EGARCH_Split_Forecast",
       # "Roll_GARCH_Forecast",
       # "Roll_EGARCH_Forecast"]

    #col = "Roll_GARCH_Forecast"

    # for debuging
    #print("NaN-Anteil:", df_dax[col].isna().mean())
    #print("<= 5*VOL_FLOOR Anteil:", (df_dax[col] <= 5*VOL_FLOOR).mean())

    #bad = df_dax.loc[(df_dax[col] <= 5*VOL_FLOOR) | (~np.isfinite(df_dax[col])), ["Date","LogReturn","HV_30d",col]]
    #print(bad.head(30))

    #col = "Roll_EGARCH_Forecast"
    #warmup = 252

    #after = df_dax.iloc[warmup:].copy()
    #print("NaN-Anteil nach warmup:", after[col].isna().mean())

    #Wo genau? (erste NaN-Stellen nach warmup)
    #print(after.loc[after[col].isna(), ["Date","LogReturn","HV_30d"]].head(30))
    
    #cols = [
    #"HV_30d","RW_Forecast","EWMA_Forecast",
    #"GARCH_Split_Forecast","EGARCH_Split_Forecast",
    #"Roll_GARCH_Forecast","Roll_EGARCH_Forecast"
    #]

    #print(df_dax[cols].max().sort_values(ascending=False).head(10))
    
    # Ploting functions for this forecast script
    """ 
    lgreturn = plot_log_returns(
        df=df,
        save_path=None,
        show=True
    )
   
    dfacf_plt = acf_plot(
        df=df,
        return_col="LogReturn",
        assetname="USD/EUR",
        save_path_r="plots/usdeur/data/usdeur_r_acf",
        save_path_r2="plots/usdeur/data/usdeur_r2_acf"
    )

    dfdistr_plot= plot_distribution(
        df=df,
        asset_name="USD/EUR",
        save_path="plots/usdeur/data/usdeur_distribution.png"
    )

    
   df_plot = plot_volatility_forecasts(
        df=df,
        forecast_specs=[
            ("RW_Forecast", "RW"),
            ("EWMA_Forecast", "EWMA"),
            ("GARCH_Split_Forecast", "Split GARCH"),
            ("EGARCH_Split_Forecast", "Split EGARCH"),
        ],
        asset_name="DAX",
        model_name="Benchmark Models",
        horizon=1,
        split_date="2018-01-01",
        oos_only=True,
        save_path=None,
        show=False
    )"""

    #df_dax[bench_cols].dropna(subset=["Date","HV_30d"]).to_csv("outputs/benchmark_forecasts.csv", index=False, sep=";")
    #df_dax[split_cols].dropna(subset=["Date","GARCH_Split_Forecast"]).to_csv("garch_split_forecasts.csv", index=False, sep=";")

    #df[all_cols].dropna(subset=["Date","HV_30d"]).to_csv("outputs/benchmark_forecasts.csv", index=False, sep=";")

    #print("Wrote benchmark_forecasts.csv")

    
    # WICHTIG
    # 1. Ordner öffnen: Masterarbeit/Python (nicht nur Datei)
    # 2. Menü: Terminal -> New Terminal
    # 3. source .venv/bin/activate
    # 4. Optional Prüfen: which python
    #   erwartet: .../Masterarbeit/Python/.venv/bin/python
    # 5. Skipt starten: python vola_pipeline_HV30.py
