import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from arch import arch_model
from tqdm import tqdm
import matplotlib.ticker as mtick
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm

from plot_forecasts import plot_volatility_forecasts
from econometric_models import load_and_prepare_data

##################################################################
# This script uses benchmark models and produces k-step-ahead
# volatility forecasts.
##################################################################

vol_floor = 1e-6

# -----------------------------------------------------------------
# helper function to clean and stabilize the forecast
# -----------------------------------------------------------------

def clip_forecast_series(x, vol_floor=vol_floor, vol_cap=None):
    """
    Clean and stabilizes a forecast series.
    Returns: a cleaned series.
    """

    # if input is already pandas series, keep it as index, so that output
    # aligns with the original data
    if isinstance(x, pd.Series):
        idx = x.index
        arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        out = pd.Series(arr, index=idx)
    else:
        # for lists or NumPy arrays, convert to numeric values first and then into pandas series
        arr = pd.to_numeric(np.asarray(x), errors="coerce").astype(float)
        out = pd.Series(arr)

    # replace positive and negative infinity with missng values
    out = out.replace([np.inf, -np.inf], np.nan)

    # enforce a minimum volatiltiy level
    out = out.clip(lower=vol_floor)

    # if an upper cap is given and finite, apply it
    if vol_cap is not None and np.isfinite(vol_cap):
        out = out.clip(upper=float(vol_cap))

    return out



# -----------------------------------------------------------------
# Random Walk and EWMA forecast
# -----------------------------------------------------------------

def add_baseline_forecast_h(df, hv_col="HV_30d", lambda_=0.94, h=21):
    """
    Adds Random Walk and EWMA forecasts.
    Needs the columns 'LogReturn' and hv_col
    """
    df = df.copy()

    # RW Forecast:: firecast akugbed to row t uses HV observed h days earlier
    df[f"RW_Forecast_h{h}"] = df[hv_col].shift(h)

    # EWMA: adjust= false gives recursive form
    alpha = 1- lambda_
    r2 = df["LogReturn"] ** 2
    ewma_var = r2.ewm(alpha=alpha, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)
    df[f"EWMA_Forecast_h{h}"] = ewma_vol.shift(h)

    col_rw = f"RW_Forecast_h{h}"
    col_ew = f"EWMA_Forecast_h{h}"

    # clean and stabilize both forecasts 
    df[col_rw] = clip_forecast_series(df[col_rw])
    df[col_ew] = clip_forecast_series(df[col_ew])

    return df



# -----------------------------------------------------------------
# Full-sample GARCH
# -----------------------------------------------------------------

def add_garch_forecast_h(
    df,
    horizons=(21, 63),
    scale=100.0,
    col_prefix="GARCH_Forecast"
):
    """
    Fit GARCH(1,1) on full sample (parametric), then produce h-step-ahead forecasts
    aligned to target dates: forecast from origin t is written into row t+h.
    """
    df = df.copy()
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    r = df["LogReturn"] * scale

    model = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant", dist="normal", rescale=False)
    res = model.fit(disp="off")

    p = res.params
    omega = float(p["omega"])
    alpha = float(p["alpha[1]"])
    beta  = float(p["beta[1]"])
    mu    = float(p["mu"])

    phi = alpha + beta
    var_floor_scaled = (vol_floor * scale) ** 2

    hv = df["HV_30d"].to_numpy(dtype=float)
    var_cap_scaled = np.nanquantile((hv * scale) ** 2, 0.999)
    if not np.isfinite(var_cap_scaled) or var_cap_scaled <= var_floor_scaled:
        # Fallback
        var_cap_scaled = np.nanquantile((r - r.mean()) ** 2, 0.999)

    # Longrun stabilisieren
    PHI_MAX = 0.995
    if (phi < PHI_MAX) and (phi > -0.999):
        longrun = omega / (1.0 - phi)
        longrun = float(np.clip(longrun, var_floor_scaled, var_cap_scaled))
    else:
        longrun = np.nan

    cond_vol = res.conditional_volatility  # scaled
    resid = res.resid                     # scaled, mean-adjusted residuals

    n = len(df)
    out = {h: np.full(n, np.nan) for h in horizons}

    # iterate over origins t; write into t+h
    for t in range(n):
        sigma_t = float(cond_vol.iloc[t])
        if not np.isfinite(sigma_t) or sigma_t <= 0:
            continue

        sigma2_t = sigma_t * sigma_t
        eps_t = float(resid.iloc[t])  # already (r_t - mu)

        # 1-step variance forecast from origin t
        sigma2_1 = omega + alpha * (eps_t ** 2) + beta * sigma2_t
        sigma2_1 = float(np.clip(sigma2_1, var_floor_scaled, var_cap_scaled))

        for h in horizons:
            target = t + h
            if target >= n:
                continue

            if np.isnan(longrun):
                sigma2_h = sigma2_1
            else:
                sigma2_h = longrun + (phi ** (h - 1)) * (sigma2_1 - longrun)

            sigma2_h = float(np.clip(sigma2_h, var_floor_scaled, var_cap_scaled))
            out[h][target] = np.sqrt(sigma2_h) / scale

    for h in horizons:
        col = f"{col_prefix}_h{h}"
        df[col] = clip_forecast_series(out[h])

    return df

def add_egarch_forecast_h(
    df,
    horizons=(21, 63),
    scale=100.0,
    col_prefix="EGARCH_Forecast"
):
    """
    Fit EGARCH(1,1) on full sample (parametric), then produce h-step-ahead forecasts
    aligned to target dates: forecast from origin t is written into row t+h.
    """
    df = df.copy()
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    r = (df["LogReturn"] * scale).astype(float)

    model = arch_model(r, vol="EGARCH", p=1, o=1, q=1, mean="Constant", dist="normal", rescale=False)
    res = model.fit(disp="off")

    p = res.params
    omega = float(p["omega"])
    beta  = float(p["beta[1]"])

    var_floor_scaled = (vol_floor * scale) ** 2

    hv = df["HV_30d"].to_numpy(dtype=float)
    var_cap_scaled = np.nanquantile((hv * scale) ** 2, 0.999)
    if not np.isfinite(var_cap_scaled) or var_cap_scaled <= var_floor_scaled:
        var_cap_scaled = np.nanquantile((r - r.mean()) ** 2, 0.999)

    # Grenzen für log-Varianz
    log_var_floor = float(np.log(var_floor_scaled))
    log_var_cap   = float(np.log(var_cap_scaled))

    cond_vol = res.conditional_volatility  # scaled

    n = len(df)
    out = {h: np.full(n, np.nan) for h in horizons}

    for t in range(n):
        sigma_t = float(cond_vol.iloc[t])
        if not np.isfinite(sigma_t) or sigma_t <= 0:
            continue

        # filtered state at origin
        log_sigma2_t = np.log(max(sigma_t * sigma_t, 1e-18))
        log_sigma2_t = float(np.clip(log_sigma2_t, log_var_floor, log_var_cap))

        for h in horizons:
            target = t + h
            if target >= n:
                continue

            # E[log sigma^2_{t+h} | F_t] under EGARCH recursion:
            if abs(1.0 - beta) < 1e-10:
                log_sigma2_h = log_sigma2_t + h * omega
            else:
                log_sigma2_h = omega * (1.0 - beta**h) / (1.0 - beta) + (beta**h) * log_sigma2_t

            log_sigma2_h = float(np.clip(log_sigma2_h, log_var_floor, log_var_cap))
            out[h][target] = np.sqrt(np.exp(log_sigma2_h)) / scale

    for h in horizons:
        col = f"{col_prefix}_h{h}"
        df[col] = clip_forecast_series(out[h])

    return df

# -----------------------------------------------------------------
# Split-sample GARCH forecast
# -----------------------------------------------------------------
def add_split_garch_forecast_h(
        df,
        split_date,
        horizons=(21, 63),
        scale=100.0,
        col_prefix="GARCH_Split_Forecast"
    ):
    """
    Fit GARCH(1,1) only on training sample then continue filtering through the test period using realized
    returns and write h-step ahead forecasts to the corresponding target dates.
    """

    df = df.copy()

    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    split_date = pd.Timestamp(split_date)

    # train period is before split date
    train_mask = df["Date"] < split_date

    # scale returns for estiamtion (as in the 1-step approach)
    returns_scaled = (df["LogReturn"] * scale).astype(float)
    returns_train = returns_scaled[train_mask]

    # lower variance bound in scaled units
    var_floor_scaled = (vol_floor * scale) ** 2

    # variance cap from historical vola
    hv = df["HV_30d"].to_numpy(dtype=float)
    var_cap_scaled = np.nanquantile((hv * scale) ** 2, 0.999)

    # fallback if the cap cannot be computed properly
    if not np.isfinite(var_cap_scaled) or var_cap_scaled <= var_floor_scaled:
        var_cap_scaled = np.nanquantile((returns_train - returns_train.mean()) ** 2, 0.999)

    # fit GARCH only on training returns
    garch = arch_model(
        returns_train,
        vol="GARCH",
        p=1, q=1,
        mean="Constant",
        dist="normal",
        rescale=False
    )

    # extract estimated parameters
    res = garch.fit(disp="off")
    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta  = float(params["beta[1]"])
    mu    = float(params["mu"])

    phi = alpha + beta
    
    # compute long-run variance only if phi<1
    PHI_MAX = 0.995
    if (phi < PHI_MAX) and (phi > -0.999):
        longrun = omega / (1.0 - phi)
        longrun = float(np.clip(longrun, var_floor_scaled, var_cap_scaled))
    else:
        longrun = np.nan

    # initialize the recursive state with last training observation
    last_train_idx = returns_train.index[-1]
    sigma_t = float(res.conditional_volatility.loc[last_train_idx])  # scaled
    sigma2_t = float(np.clip(sigma_t * sigma_t, var_floor_scaled, var_cap_scaled))
    eps_t = float(returns_scaled.loc[last_train_idx] - mu)

    n = len(df)
    out = {h: np.full(n, np.nan) for h in horizons}

    # since dataframe index has been reset, the last training index is used as positional index
    pos_last_train = int(last_train_idx)  

    # move forward through the test period one day at a time
    for pos in range(pos_last_train + 1, n):
        # one-step variance forecast for the current day based on yesterdays state
        sigma2_next = omega + alpha * (eps_t ** 2) + beta * sigma2_t
        sigma2_next = float(np.clip(sigma2_next, var_floor_scaled, var_cap_scaled))

        # update the state using todays reallizd return
        eps_t = float(returns_scaled.iloc[pos] - mu)
        sigma2_t = sigma2_next

        # from now on: current date at the new forecast origin
        # compute one-step-ahead variance from todays updated state
        sigma2_1_ahead = omega + alpha * (eps_t ** 2) + beta * sigma2_t
        sigma2_1_ahead = float(np.clip(sigma2_1_ahead, var_floor_scaled, var_cap_scaled))

        # Ffor each horizon h, write the forecast to the target date position+h
        for h in horizons:
            h = int(h)
            target_pos = pos + h
            if target_pos >= n:
                continue
            
            # if no stable long-run variance available --> use one-step-ahead forecast as fallback
            if np.isnan(longrun):
                sigma2_h = sigma2_1_ahead
            else:
                sigma2_h = longrun + (phi ** (h - 1)) * (sigma2_1_ahead - longrun)

            sigma2_h = float(np.clip(sigma2_h, var_floor_scaled, var_cap_scaled))
            out[h][target_pos] = np.sqrt(sigma2_h) / scale

    # attach the cleaned forecast columns
    for h in horizons:
        h = int(h)
        col = f"{col_prefix}_h{h}"
        df[col] = clip_forecast_series(out[h])

    return df


def add_split_egarch_forecast_h(
        df,
        split_date,
        horizons=(21, 63),
        scale=100.0,
        col_prefix="EGARCH_Split_Forecast"
    ):
    """
    Again similar to the one-step-ahead forecast, EGARCH only on training sample,
    then fixing parameters and produce the h-step-ahead forecasts.
    """

    df = df.copy()

    # ensure date is datetimem and sorted
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    split_date = pd.Timestamp(split_date)
    train_mask = df["Date"] < split_date

    # scale returns for estiamtion
    returns_scaled = (df["LogReturn"] * scale).astype(float)
    returns_train = returns_scaled[train_mask]

    # variance bounds
    var_floor_scaled = (vol_floor * scale) ** 2

    hv = df["HV_30d"].to_numpy(dtype=float)
    var_cap_scaled = np.nanquantile((hv * scale) ** 2, 0.999)
    if not np.isfinite(var_cap_scaled) or var_cap_scaled <= var_floor_scaled:
        var_cap_scaled = np.nanquantile((returns_train - returns_train.mean()) ** 2, 0.999)

    # convert variance bounds to log-variance bounds
    log_var_floor = float(np.log(var_floor_scaled))
    log_var_cap   = float(np.log(var_cap_scaled))

    # fit EGARCH only on the training period
    egarch = arch_model(
        returns_train,
        vol="EGARCH",
        p=1, o=1, q=1,
        mean="Constant",
        dist="normal",
        rescale=False
    )
    res = egarch.fit(disp="off")
    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta  = float(params["beta[1]"])
    gamma = float(params["gamma[1]"])
    mu    = float(params["mu"])

    # E|z| for standard normal z
    E_abs_z = np.sqrt(2.0 / np.pi)

    # initialize the recursion on last training day
    last_train_idx = returns_train.index[-1]

    sigma_t = float(res.conditional_volatility.loc[last_train_idx])  # scaled
    sigma2_t = float(np.clip(sigma_t * sigma_t, var_floor_scaled, var_cap_scaled))
    log_sigma2_t = float(np.clip(np.log(sigma2_t), log_var_floor, log_var_cap))

    eps_t = float(returns_scaled.loc[last_train_idx] - mu)
    sigma_t_safe = float(np.sqrt(np.exp(log_sigma2_t)))
    z_t = float(eps_t / max(sigma_t_safe, 1e-18))

    n = len(df)
    out = {int(h): np.full(n, np.nan, dtype=float) for h in horizons}

    pos_last_train = int(last_train_idx)

    # step through the test sample and keep updating filtered state
    for pos in range(pos_last_train + 1, n):
        # update todays log variance using yesterdays state
        log_sigma2_next = (
            omega
            + beta * log_sigma2_t
            + alpha * (abs(z_t) - E_abs_z)
            + gamma * z_t
        )
        log_sigma2_next = float(np.clip(log_sigma2_next, log_var_floor, log_var_cap)) # scaled
        sigma_next = float(np.sqrt(np.exp(log_sigma2_next))) 

        # use todays realized return to construct todays standardized shock
        eps_next = float(returns_scaled.iloc[pos] - mu)
        z_next = float(eps_next / max(sigma_next, 1e-18))

        # set the updated state as the new current state
        log_sigma2_t = log_sigma2_next
        z_t = z_next

        # produce h-step forecasts from todays origin
        for h in horizons:
            h = int(h)
            target_pos = pos + h
            if target_pos >= n:
                continue

            # expected future log variance
            if abs(1.0 - beta) < 1e-10:
                log_sigma2_h = log_sigma2_t + h * omega
            else:
                log_sigma2_h = omega * (1.0 - beta**h) / (1.0 - beta) + (beta**h) * log_sigma2_t

            log_sigma2_h = float(np.clip(log_sigma2_h, log_var_floor, log_var_cap))
            out[h][target_pos] = np.sqrt(np.exp(log_sigma2_h)) / scale

    for h in horizons:
        h = int(h)
        col = f"{col_prefix}_h{h}"
        df[col] = clip_forecast_series(out[h])

    return df

# --------------------------------------------------
# Rolling GARCH and EGARCH forecast
# --------------------------------------------------

def add_rolling_garch_forecast_h(
    df,
    window=252,
    horizons=(21, 63),
    scale=100.0,
    col_prefix="Roll_GARCH_Forecast",
    mean: str = "Zero",
    dist: str = "normal",
    aggregation: str = "point_vol",  
):
    """
    Produce rolling-window GARCH(1,1) horizon forecasts.

    For each forecast origin t:
    1. fit GARCH on the most recent rolling window of returns.
    2. forecast variance path up to the maximum requested horizon.
    3. convert that variance path into horizon-specific forecasts.
    4. write each forecast into row t+h.
    """

    df = df.copy()

    # input checks
    if "LogReturn" not in df.columns:
        raise ValueError("DataFrame must contain a 'LogReturn' column")

    if window is None or window <= 10:
        raise ValueError("'window' must be an integer > 10")

    horizons = tuple(int(h) for h in horizons)
    if any(h <= 0 for h in horizons):
        raise ValueError("'horizons' must contain positive integers")

    if aggregation not in {"cumulative_vol", "point_vol"}:
        raise ValueError("aggregation must be 'cumulative_vol' or 'point_vol'")

    # scale returns for numerical stability
    returns_scaled = (df["LogReturn"] * scale).astype(float)
    n = len(returns_scaled)

    # lower and upper variance bound in scaled units
    var_floor_scaled = (vol_floor * scale) ** 2
    hv = df["HV_30d"].to_numpy(dtype=float)
    var_cap_scaled = np.nanquantile((hv * scale) ** 2, 0.999)
    if not np.isfinite(var_cap_scaled) or var_cap_scaled <= var_floor_scaled:
        var_cap_scaled = np.nanquantile((returns_scaled - np.nanmean(returns_scaled)) ** 2, 0.999)

    # Prepare result arrays for each horizon
    out = {h: np.full(n, np.nan, dtype=float) for h in horizons}

    # valid forecast origins star once the first full rolling window is available
    start_i = window - 1

    # ensure that i+h never exceeds the dataset length
    max_h = max(horizons)
    end_i = n - 1 - max_h  

    # loop through all rolling forecast origins
    for i in tqdm(range(start_i, end_i + 1), desc="Rolling GARCH (horizon)"):
        # extract the rolling estimated window ending at i
        window_returns = returns_scaled.iloc[i - window + 1: i + 1]

        # skip this window if it contains invalid numbers
        if not np.isfinite(window_returns.values).all():
            continue

        try:
            # fit GARCH on rolling window
            garch = arch_model(
                window_returns,
                vol="GARCH",
                p=1,
                q=1,
                mean=mean,
                dist=dist,
                rescale=False
            )
            res = garch.fit(disp="off", show_warning=False)

            # forecast up to the maximum horizon once, reuse for all h
            fcast = res.forecast(horizon=max_h, reindex=False)

            # take forecast path associated with the last row of estimation window
            var_path_scaled = np.asarray(fcast.variance.values[-1, :], dtype=float)

            # skip invalid variance paths
            if (not np.isfinite(var_path_scaled).all()) or np.any(var_path_scaled < 0):
                continue
            
            # stabilize the path by clipping extreme values
            var_path_scaled = np.clip(var_path_scaled, var_floor_scaled, var_cap_scaled)   

            # create and store forecasts for each requested horizon
            for h in horizons:
                if aggregation == "cumulative_vol":
                    # sum future variance over the next h days and take the square root
                    var_h_scaled = float(var_path_scaled[:h].sum())
                else:
                    # take only the h-th step-ahead variance
                    var_h_scaled = float(var_path_scaled[h - 1])

                if not np.isfinite(var_h_scaled) or var_h_scaled < var_floor_scaled:
                    continue

                vol_h = np.sqrt(var_h_scaled) / scale

                # write to time t+h
                out[h][i + h] = vol_h

        # if a fit fails for one window, skip that window
        except Exception:
            continue

    for h in horizons:
        col = f"{col_prefix}_h{h}"
        df[col] = out[h]                      
        df[col] = clip_forecast_series(df[col])


    return df



def add_rolling_egarch_forecast_h(
    df,
    window=252,
    horizons=(21, 63),
    scale=10.0,
    col_prefix="Roll_EGARCH_Forecast",
    mean: str = "Constant",
    dist: str = "t",
    aggregation: str = "point_vol",  # "cumulative_vol" or "point_vol"
    simulations: int = 1000,
    seed: int = 42,
    max_bad_frac: float = 0.05,
    use_median: bool = False,        # median across sims if simulation arrays accessible
):
    
    """
    Produce rolling-window EGARCH horizon forecasts using simulation-based forecasts.

    For each forecast origin:
    1. fit EGARCH on the current rolling window.
    2. simulate a future variance path.
    3. aggregate the path depending on the selected horizon definition.
    4. store the result in the target row.
    
    """

    df = df.copy()

    # input columns
    if "LogReturn" not in df.columns:
        raise ValueError("DataFrame must contain a 'LogReturn' column")
    if "HV_30d" not in df.columns:
        raise ValueError("DataFrame must contain an 'HV_30d' column")

    horizons = tuple(int(h) for h in horizons)
    if aggregation not in {"cumulative_vol", "point_vol"}:
        raise ValueError("aggregation must be 'cumulative_vol' or 'point_vol'")

    # scale returns for fitting
    returns_scaled = (df["LogReturn"].astype(float) * float(scale))
    n = len(returns_scaled)

    # lower bound for scaled variance
    var_floor_scaled = (vol_floor * scale) ** 2

    # build an upper cap from level of scaled historical vola
    hv_scaled = (df["HV_30d"].to_numpy(float) * scale)
    typ = np.nanmedian(hv_scaled)
    var_cap_scaled = (4.0 * typ) ** 2

    # fallback if cap is invalid
    if not np.isfinite(var_cap_scaled) or var_cap_scaled <= var_floor_scaled:
        var_cap_scaled = np.nanquantile((returns_scaled - np.nanmean(returns_scaled)) ** 2, 0.999)

    out = {h: np.full(n, np.nan, dtype=float) for h in horizons}

    # earliest possible forecast origin
    start_i = window - 1
    max_h = max(horizons)
    end_i = n - 1 - max_h

    # if sample is too short for rolling forecating, create empty forecast columns
    # and return immediately
    if end_i < start_i:
        for h in horizons:
            df[f"{col_prefix}_h{h}"] = np.nan
        return df

    # diagnostic counters
    success = 0
    fail_fit = 0
    fail_forecast = 0
    fail_badpath = 0
    first_err = None

    # loop over all rolling forecast origins
    for i in tqdm(range(start_i, end_i + 1), desc="Rolling EGARCH (simulation)"):
        window_returns = returns_scaled.iloc[i - window + 1: i + 1]

        # skip windows with invalid returns
        if not np.isfinite(window_returns.values).all():
            continue

        try:
            egarch = arch_model(
                window_returns,
                vol="EGARCH",
                p=1, o=1, q=1,
                mean=mean,
                dist=dist,
                rescale=False
            )
            res = egarch.fit(disp="off", show_warning=False, options={"maxiter": 2000})
        except Exception as e:
            fail_fit += 1
            if first_err is None:
                first_err = ("FIT", i, repr(e))
            continue

        try:
            # use a different random sead for each origin, reduces artifical similarity across
            # forecast origins
            rng = np.random.default_rng(seed + i)

            # simulation based forecast path up to max_h
            fcast = res.forecast(
                horizon=max_h,
                reindex=False,
                method="simulation",
                simulations=simulations,
                rng=rng.standard_normal,  # callable(size)->array
            )

            # prefer version-stable output: expected variance path
            var_path_scaled = np.asarray(fcast.variance.values[-1, :], dtype=float)

            # optional robust aggregation across simulation paths (if accessible)
            if use_median:
                sims = getattr(fcast, "simulations", None)
                if sims is not None:
                    # try common attribute names
                    for attr in ("variances", "variance"):
                        if hasattr(sims, attr):
                            var_sims = np.asarray(getattr(sims, attr))
                            # expected shape often: (t, simulations, horizon)
                            # take last origin row:
                            var_sims = var_sims[-1]
                            # now (simulations, horizon)
                            if var_sims.ndim == 2:
                                var_path_scaled = np.nanmedian(var_sims, axis=0)
                            break
            
            # skip invalid forecat paths
            if (not np.isfinite(var_path_scaled).all()) or np.any(var_path_scaled < 0):
                fail_badpath += 1
                continue    

            # enforce rasonable bounds
            var_path_scaled = np.clip(var_path_scaled, var_floor_scaled, var_cap_scaled)

        except Exception as e:
            fail_forecast += 1
            if first_err is None:
                first_err = ("FORECAST", i, repr(e))
            continue

        # check wether too many elemnts of the path are invalied
        ok = np.isfinite(var_path_scaled) & (var_path_scaled >= 0)
        bad_frac = 1.0 - ok.mean()
        if bad_frac > max_bad_frac or ok.sum() == 0:
            fail_badpath += 1
            continue

        # write forecasts for all requested horizons
        for h in horizons:
            if aggregation == "cumulative_vol":
                var_h_scaled = float(np.sum(var_path_scaled[:h]))
            else:
                var_h_scaled = float(var_path_scaled[h - 1])

            if not np.isfinite(var_h_scaled) or var_h_scaled < var_floor_scaled:
                continue
            out[h][i + h] = np.sqrt(var_h_scaled) / scale

        success += 1

    for h in horizons:
        col = f"{col_prefix}_h{h}"
        df[col] = clip_forecast_series(out[h], vol_floor=vol_floor)

    return df


#############################################################################################################
## MAIN
#############################################################################################################

if __name__ == "__main__":
    horizons = (21,63)
    split_date = "2018-01-01"
    
    df = load_and_prepare_data(
        path = "USD_EUR Historical Data.csv",
        sep=";",
        decimal=",",
        date_format="%m.%d.%Y",
        drop_cols=["Open", "High", "Low", "Vol,", "Change %"]
    )

   
    for h in [21, 63]:
        # add RW and EWMA
        df = add_baseline_forecast_h(df, h=h)

    # split-sample GARCH and EGARCH
    df = add_split_garch_forecast_h(df, split_date="2018-01-01", horizons=(21,63))
    df = add_split_egarch_forecast_h(df, split_date="2018-01-01", horizons=(21,63))

    # rolling window GARCH and EGARCH
    #df = add_rolling_garch_forecast_h(df, window=252, horizons=(21,63), col_prefix="Roll_GARCH_Forecast", mean="Zero", dist="normal")
    #df = add_rolling_egarch_forecast_h( df=df, window=252,horizons=(21, 63),col_prefix="Roll_EGARCH_Forecast",mean="Constant",dist="t",aggregation="point_vol",     # or "cumulative_vol"
    #scale=10.0,simulations=1000,seed=42)
    
    # column selection for exporting
    all_cols = [
        "Date",
        "LogReturn",
        "HV_30d",
        "RW_Forecast_h21", "EWMA_Forecast_h21",
        "RW_Forecast_h63", "EWMA_Forecast_h63",
        "GARCH_Split_Forecast_h21",
        "GARCH_Split_Forecast_h63",
        "EGARCH_Split_Forecast_h21",
        "EGARCH_Split_Forecast_h63",
        #"Roll_GARCH_Forecast_h21",
        #"Roll_GARCH_Forecast_h63",
        #"Roll_EGARCH_Forecast_h21",
        #"Roll_EGARCH_Forecast_h63"
    ] 

    plot_volatility_forecasts(
        df=df,
        forecast_specs=[
            #("RW_Forecast_h21", "RW"),
            #("EWMA_Forecast_h21", "EWMA"),
            #("GARCH_Split_Forecast_h21", "GARCH (Split)"),
            #("EGARCH_Split_Forecast_h21", "EGARCH (Split)"),
            #("Roll_GARCH_Forecast_h21", "Roll GARCH"),
            ("Roll_EGARCH_Forecast_h21", "Roll EGARCH"),
        ],
        asset_name="USD/EUR",
        model_name="Benchmark Models",
        split_date="2018-01-01",
        oos_only=True,
        horizon=21,
        show=True,
        save_path="plots/usdeur_rollegarchbenchmark_h21"
    )
    
    plot_volatility_forecasts(
        df=df,
        forecast_specs=[
            #("RW_Forecast_h63", "RW"),
            #("EWMA_Forecast_h63", "EWMA"),
            #("GARCH_Split_Forecast_h63", "Split GARCH"),
            #("EGARCH_Split_Forecast_h63", "Split EGARCH"),
            #("Roll_GARCH_Forecast_h63", "Roll GARCH"),
            ("Roll_EGARCH_Forecast_h63", "Roll EGARCH"),
            
        ],
        asset_name="USD/EUR",
        model_name="Benchmark Models",
        split_date="2018-01-01",
        oos_only=True,
        horizon=63,
        show=True,
        save_path="plots/usdeur_rollegarchbenchmark_h63"
    )
    
    # export commands
    #df[all_cols].dropna(subset=["Date", "HV_30d"]).to_csv("outputs/benchmark_forecasts_h.csv", index=False, sep=";")
    #print("Wrote benchmark_forecasts_h.csv")

    
    # WICHTIG
    # 1. Ordner öffnen: Masterarbeit/Python (nicht nur Datei)
    # 2. Menü: Terminal -> New Terminal
    # 3. source .venv/bin/activate
    # 4. Optional Prüfen: which python
    #   erwartet: .../Masterarbeit/Python/.venv/bin/python
    # 5. Skipt starten: python vola_pipeline_HV30.py

    # DM -test
    # python evaluate_all_models_h.py --horizons 21 --dm --dm-loss qlike --dm-baseline "GARCH (split)"
    # or
    # python evaluate_all_models_h.py --horizons 63 --dm --dm-loss qlike --dm-baseline "G-LSTM"