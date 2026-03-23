import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from tqdm import tqdm
import matplotlib.ticker as mtick
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from plot_forecasts import plot_volatility_forecasts    


VOL_FLOOR = 1e-6

# ----
# ---- Evaluation Functions ----
# ----

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def qlike(y_true, y_pred, var_floor=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    var_true = y_true**2
    var_pred = y_pred**2

    # floor at variance level
    var_pred = np.clip(var_pred, var_floor, None)

    out = np.log(var_pred) + var_true / var_pred
    out = out[np.isfinite(out)]
    return float(np.mean(out))

# ----
# ---- Dataextraction ----
# ----


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
    Liest Datenreihe ein, berechnet LogReturns und HV_30d. 
    Gibt ein DataFrame mit Spalten [Date, Price, LogReturn, HV_30d] zurück
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


def add_vdax_data(
        df,
        path,
        sep,
        decimal,
        date_format,
        drop_cols,
        price_col="Price",
        date_col="Date"
):
    
    df = df.copy()

    vdf = pd.read_csv(
        path,
        sep=sep,
        dtype={date_col: str},
        decimal=decimal
    )

    if drop_cols:
        vdf = vdf.drop(columns=drop_cols)
    
    vdf[date_col] = pd.to_datetime(vdf[date_col], format=date_format)
    vdf = vdf.sort_values(date_col)

    vdf[price_col] = vdf[price_col] / (np.sqrt(252) * 100)

    vdf = vdf[[date_col, price_col]].rename(columns={price_col: "VDAX"})

    df[date_col] = pd.to_datetime(df[date_col])
    vdf = vdf.drop_duplicates(subset=[date_col], keep="last")
    df = df.merge(vdf, on=date_col, validate="one_to_one", how="left")
    
    return df


def give_descriptive_stats(series):
    "Function: Basic descriptive stats used in Section 2.3 (returns & volatility)."
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
# ----
# ---- RW and EWMA ----
# ----

def add_baseline_forecast(df, hv_col="HV_30d", lambda_=0.94):
    """
    Adds Random Walk and EWMA forecasts.
    Needs the columns 'LogReturn' and hv_col
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


# ----
# ---- GARCH ----
# ----

# return_model = False: return the dataframe with added columns
# return_model = True: return a tuple (df, return_model); for access to the fitted model object, to do something like model.summary()
def add_garch_forecast(df, scale=100.0, return_model=False):
    df = df.copy()
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

    # WICHTIG: reindex=True + start=0 => Forecast-Serie über die ganze Stichprobe
    fcast = garch_res.forecast(horizon=1, start=0, reindex=True)

    # 1-step-ahead Varianz (skaliert), als Series aligned zum Index
    var_1step_scaled = fcast.variance["h.1"].copy()
    var_1step_scaled = var_1step_scaled.clip(lower=VAR_FLOOR_SCALED)

    vol_1step = np.sqrt(var_1step_scaled) / scale

    # optional: deine Konvention "Forecast für t kommt aus Info bis t-1"
    df["GARCH_Forecast"] = (
        vol_1step.shift(1)
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=VOL_FLOOR)
    )

    if return_model:
        return df, garch_res
    return df

def add_egarch_forecast(df, scale=100.0, return_model=False):
    df = df.copy()
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

    fcast = egarch_res.forecast(horizon=1, start=0, reindex=True)
    var_1step_scaled = fcast.variance["h.1"].copy()
    var_1step_scaled = var_1step_scaled.clip(lower=VAR_FLOOR_SCALED)

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
    Schätzt ein GARCH(1,1) nur auf Daten VOR split_date (Train)
    und erzeugt dann ab split_date echte 1-Step-ahead-Volatilitäts-
    forecasts auf Basis der geschätzten Parameter.

    - df: DataFrame mit Spalten ["Date", "LogReturn", ...]
    - split_date: z.B. "2018-01-01"
    - forecast_col: Name der Spalte für die Vorhersage (Standard: "GARCH_Split_Forecast")
    """

    df = df.copy()

    # Sicherstellen, dass Date eine Datetime-Spalte ist
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    # Nach Datum sortieren (solltest du eh schon haben)
    df = df.sort_values("Date").reset_index(drop=True)

    split_date = pd.Timestamp(split_date)

    # Maske Train/Test
    # Boolean-Serie, True für alle Werte die in _mask dabei sind
    train_mask = df["Date"] < split_date
    test_mask  = df["Date"] >= split_date

    # Skalierte Returns (wie in add_garch_forecast)
    returns_scaled = df["LogReturn"] * scale

    # Nur Train-Returns zum Schätzen
    # Filtert nur LogRetorns aus train_mask
    returns_train = returns_scaled[train_mask]

    # GARCH(1,1) auf Train schätzen
    garch = arch_model(
        returns_train,
        vol="GARCH",
        p=1,
        q=1,
        mean="Constant",
        dist="normal"
    )

    # Schätzt die Parameter omega, alpha, beta, mü mit Maximum Likehihood
    res = garch.fit(disp="off")

    # Geschätzte parameter holen
    params = res.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta  = params["beta[1]"]
    mu    = params["mu"]   # Konstanten-Mean

    # Letzte Train-Zeit t0: Tag vor dem Split -> Startpunkt fürs Weiterschreiben der GARCH Rekursion
    last_train_idx = returns_train.index[-1]

    # Bedingte Volatilität zum letzten Train-Zeitpunkt (skaliert)
    sigma_t = res.conditional_volatility.loc[last_train_idx]  # σ_t (scaled)
    sigma2_t = sigma_t ** 2

    # eps_t = r_t - m
    # Residuum epsilon_t zum letzten Train-Zeitpunkt (skaliert)
    eps_t = returns_scaled.loc[last_train_idx] - mu

    # Forecast-Array vorbereiten (Länge wie df), füllen es mit Nan
    garch_forecast = np.full(len(df), np.nan)

    # Wir gehen ab dem Tag NACH last_train_idx vorwärts durch die Zeit
    # und berechnen jeweils σ_{t+1}^2 = ω + α * ε_t^2 + β * σ_t^2
    # Der Forecast für das Datum t+1 ist dann σ_{t+1} / scale
    full_index = df.index

    # Position von last_train_idx im DataFrame
    pos_last_train = full_index.get_loc(last_train_idx)
    VAR_FLOOR_SCALED = (VOL_FLOOR * scale) ** 2

    # Schleife von dem Tag nach dem letzten Train-Tag bis zum Ende
    for pos in range(pos_last_train + 1, len(full_index)):
        idx = full_index[pos]      # aktuelles Datum t+1 im df

        sigma2_next = omega + alpha * (eps_t ** 2) + beta * sigma2_t
        if not np.isfinite(sigma2_next):
            sigma2_next = VAR_FLOOR_SCALED
        sigma2_next = max(float(sigma2_next), VAR_FLOOR_SCALED)

        sigma_next = np.sqrt(sigma2_next)
        garch_forecast[pos] = max(float(sigma_next / scale), VOL_FLOOR)

        # Für den nächsten Schritt: jetzt ist dies t, wir brauchen eps_t und sigma2_t
        # eps_t basiert auf dem REALISIERTEN Return am heutigen Datum idx
        eps_t   = returns_scaled.loc[idx] - mu
        sigma2_t = sigma2_next

    # Forecast-Spalte ins df schreiben
    df[forecast_col] = garch_forecast

    return df

def add_split_egarch_forecast(
    df,
    split_date,
    scale=100.0,
    forecast_col="EGARCH_Split_Forecast",
    cap_quantile=0.999
):
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
    var_cap_scaled = np.nanquantile((hv * scale) ** 2, cap_quantile)
    if (not np.isfinite(var_cap_scaled)) or (var_cap_scaled <= var_floor_scaled):
        var_cap_scaled = np.nanquantile((returns_train - np.nanmean(returns_train)) ** 2, cap_quantile)

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

    # --- Startzustand: letzter Train-Tag ---
    last_train_idx = returns_train.index[-1]
    sigma_t = float(res.conditional_volatility.loc[last_train_idx])  # scaled
    sigma2_t = float(np.clip(sigma_t * sigma_t, var_floor_scaled, var_cap_scaled))
    log_sigma2_t = float(np.clip(np.log(sigma2_t), log_var_floor, log_var_cap))

    eps_t = float(returns_scaled.loc[last_train_idx] - mu)

    # Output
    out = np.full(len(df), np.nan, dtype=float)

    pos_last_train = int(last_train_idx)  # nach reset_index passt das
    for pos in range(pos_last_train + 1, len(df)):
        # sigma_t (scaled) aus log-state
        sigma_t = float(np.sqrt(np.exp(log_sigma2_t)))
        sigma_t = max(sigma_t, float(np.sqrt(var_floor_scaled)))

        # z_t
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

        # --- entscheidend: clip in log-space ---
        log_sigma2_next = float(np.clip(log_sigma2_next, log_var_floor, log_var_cap))

        # 1-step-ahead forecast (für "heute" pos) in Originalskalierung
        vol_next = float(np.sqrt(np.exp(log_sigma2_next)) / scale)
        out[pos] = max(vol_next, VOL_FLOOR)

        # Zustand updaten mit realisiertem Return von "heute"
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

"""
def add_rolling_garch_forecast(
    df,
    window=252,
    scale=100.0,
    forecast_col="Roll_GARCH_Forecast",
    mean: str = "Constant",          # <-- change
    dist: str = "normal",
    cap_quantile: float = 0.999,
    force_fill: bool = True,         # like EGARCH: fill gaps after first value
):
    df = df.copy()

    if "LogReturn" not in df.columns:
        raise ValueError("DataFrame must contain a 'LogReturn' column")
    if window is None or window <= 10:
        raise ValueError("'window' must be an integer > 10")

    r = (df["LogReturn"] * scale).astype(float)
    n = len(r)
    out = np.full(n, np.nan, dtype=float)

    # variance floor/cap in scaled units
    var_floor_scaled = (VOL_FLOOR * scale) ** 2

    if "HV_30d" in df.columns:
        hv = df["HV_30d"].to_numpy(dtype=float)
        var_cap_scaled = np.nanquantile((hv * scale) ** 2, cap_quantile)
    else:
        var_cap_scaled = np.nanquantile((r - np.nanmean(r)) ** 2, cap_quantile)

    if (not np.isfinite(var_cap_scaled)) or (var_cap_scaled <= var_floor_scaled):
        var_cap_scaled = np.nanquantile((r - np.nanmean(r)) ** 2, 0.999)

    def carry_forward(j: int):
        if j > 0 and np.isfinite(out[j - 1]):
            out[j] = float(out[j - 1])

    start_i = window - 1
    end_i = n - 2  # we write into i+1

    for i in tqdm(range(start_i, end_i + 1), desc="Rolling GARCH"):
        j = i + 1
        window_r = r.iloc[i - window + 1: i + 1]

        if not np.isfinite(window_r.values).all():
            carry_forward(j)
            continue

        try:
            garch = arch_model(
                window_r,
                vol="GARCH",
                p=1, q=1,
                mean=mean,
                dist=dist,
                rescale=False
            )
            res = garch.fit(disp="off", show_warning=False)

            # 1) convergence filter
            if getattr(res, "convergence_flag", 0) != 0:
                carry_forward(j)
                continue

            # 2) persistence filter (alpha+beta too close to 1 => unstable)
            p = getattr(res, "params", None)
            if p is not None and ("alpha[1]" in p) and ("beta[1]" in p):
                phi = float(p["alpha[1]"]) + float(p["beta[1]"])
                if not np.isfinite(phi) or phi > 0.9995:
                    carry_forward(j)
                    continue

            fcast = res.forecast(horizon=1, reindex=False)
            var_next_scaled = float(fcast.variance.values[-1, 0])

            if (not np.isfinite(var_next_scaled)) or (var_next_scaled <= 0):
                carry_forward(j)
                continue

            # 3) variance clipping
            var_next_scaled = float(np.clip(var_next_scaled, var_floor_scaled, var_cap_scaled))

            vol_next = float(np.sqrt(var_next_scaled) / scale)
            vol_next = max(vol_next, VOL_FLOOR)
            out[j] = vol_next

        except Exception:
            carry_forward(j)
            continue

    s = pd.Series(out, index=df.index)
    if force_fill:
        s = s.ffill()

    s = s.replace([np.inf, -np.inf], np.nan).clip(lower=VOL_FLOOR)
    df[forecast_col] = s
    return df

"""

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
    df = df.copy()

    if "LogReturn" not in df.columns:
        raise ValueError("Dataframe needs the column 'LogReturn'")
    if window is None or window <= 10:
        raise ValueError("'window' must be an integer > 10")

    r = (df["LogReturn"] * scale).astype(float)
    n = len(r)
    out = np.full(n, np.nan, dtype=float)

    var_floor_scaled = (VOL_FLOOR * scale) ** 2

    # datenbasiertes Cap (wie bei h=21/63)
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
        # nimm letzten validen Forecast, falls vorhanden
        if j > 0 and np.isfinite(out[j - 1]):
            out[j] = float(out[j - 1])

    start_i = window - 1
    end_i = n - 2  # wir schreiben in i+1

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

            # Konvergenz prüfen (viele "Ausreißer" kommen von bad fits)
            if getattr(res, "convergence_flag", 0) != 0:
                carry_forward(j)
                continue

            # Extrem persistente Fits tendieren zu instabilen 1-step forecasts
            params = getattr(res, "params", None)
            if params is not None and "beta[1]" in params and float(params["beta[1]"]) > 0.9995:
                carry_forward(j)
                continue

            fcast = res.forecast(horizon=1, reindex=False)
            var_next_scaled = float(fcast.variance.values[-1, 0])

            if (not np.isfinite(var_next_scaled)) or (var_next_scaled <= 0):
                carry_forward(j)
                continue

            # EGARCH ist log-Varianz-getrieben: clippe sinnvoll in log-space
            log_var_next = float(np.log(var_next_scaled))
            log_var_next = float(np.clip(log_var_next, log_var_floor, log_var_cap))
            var_next_scaled = float(np.exp(log_var_next))

            vol_next = float(np.sqrt(var_next_scaled) / scale)
            vol_next = max(vol_next, VOL_FLOOR)

            # optionales Smoothing: reduziert "Zappeln" bei h=1
            if smooth_alpha and smooth_alpha > 0 and j > 0 and np.isfinite(out[j - 1]):
                vol_next = float(smooth_alpha * vol_next + (1.0 - smooth_alpha) * out[j - 1])

            out[j] = vol_next

        except Exception:
            carry_forward(j)
            continue

    s = pd.Series(out, index=df.index)

    if force_fill:
        # sobald der erste Forecast da ist: Lücken durchziehen
        s = s.ffill()

    # finaler Clip (nur Floor, Cap ist bereits im log-clip drin, aber schadet nicht)
    s = s.replace([np.inf, -np.inf], np.nan).clip(lower=VOL_FLOOR)

    df[forecast_col] = s
    return df
# ----------------
# ---- Plot-Functions ----
# ----------------

def plot_vola_forecasts(
    df,
    title,
    hv_col="HV_30d",
    columns_to_plot=None,
    date_col="Date",
    start="2018-01-01"
):
    """
    Plots the hsitorical volatility and chosen forecast-columns.
    columns_to_plot: List of (Columnname, Lable) tuples.
    """

    df = df.copy()

    if columns_to_plot is None:
            columns_to_plot = []
    
    # Set Date as index (if part of df.columns, what is the case for all my dataframes, but if not then the
    # x axis would be indexed as 1, 2,...
    if date_col in df.columns:
        df= df.set_index(date_col)

    if start is not None:
        start = pd.Timestamp(start)
        df = df.loc[df.index >= start]

    plt.figure(figsize=(12,6))
    plt.plot(df[hv_col], label=f"Historical Volatility ({hv_col})", linewidth=2, color="black")

    for col, label in columns_to_plot:
        if col in df.columns:
            plt.plot(df[col], label=label, linewidth=1)
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    
    plt.subplots_adjust(
        top=0.94,
        bottom=0.08,
        left=0.06,
        right=0.98,
        hspace=0.2,
        wspace=0.2
    )
    plt.show()

    return df


def plot_distribution(
        df, 
        title,
        return_col="LogReturn"):
    
    r = df[return_col].dropna()

    mu = r.mean()
    sigma = r.std(ddof=0)

    fig, ax = plt.subplots(figsize=(9,5))

    ax.hist(
        r,
        bins=80,
        density=True,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.95,
        label="Empirical distribution"
    )

    x = np.linspace(r.min(), r.max(), 500)
    pdf = norm.pdf(x, loc=mu, scale=sigma)

    ax.plot(
        x,
        pdf,
        color="orange",
        linestyle="-",
        linewidth=1.2,
        label="Normal distribution"
    )

    ax.axvline(r.mean(), linewidth=1.2, color="black", linestyle=":", label="Mean")

    ax.set_title(title)
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")

    # Achsenformatierung (r in %)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))

    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    #plt.show()

def acf_plot(
        df,
        return_col="LogReturn",
        assetname="Name"
):
    df = df.copy()
    r = df["LogReturn"]
    r2 = r**2

    plot_acf(r, lags=50, title=f"{assetname} LogReturn Autocorrelation")
    plot_acf(r2, lags=50, title=rf"{assetname} $\text{{LogReturn}}^2$  Autocorrelation")

    plt.show()


#############################################################################################################
# ---- Für mich
if __name__ == "__main__":

    df = load_and_prepare_data(
        path = "USD_EUR Historical Data.csv",
        sep=";",
        decimal=",",
        date_format="%m.%d.%Y",
        drop_cols=["Open", "High", "Vol,", "Change %"]
    )


    df = add_baseline_forecast(df=df)
    df = add_garch_forecast(df=df)
    df = add_egarch_forecast(df=df)
    df = add_split_garch_forecast(df, split_date="2018-01-01")
    df = add_split_egarch_forecast(df=df, split_date="2018-01-01")
    #df = add_rolling_garch_forecast(
     #   df = df,
     #   window= 252,
     #   scale=100.0,
     #   forecast_col= "Roll_GARCH_Forecast",
     #   mean = "Zero",
     #   dist ="normal",
        #cap_quantile=0.999,
        #force_fill=True 
      # )
    
    #df = add_rolling_egarch(
     #   df=df,
     #   window=252,
     #   scale=100.0,
      #  forecast_col="Roll_EGARCH_Forecast",
      #  mean="Zero",
      #  dist="normal",
      #  smooth_alpha=0.2)
    
    bench_cols = [
        "Date",
        "HV_30d",
        "RW_Forecast",
        "EWMA_Forecast"
    ]

    split_cols = [
        "Date",
        "GARCH_Split_Forecast"
    ]

    roll_cols = [
        "Date",
        #"Roll_GARCH_Forecast",
        #"Roll_EGARCH_Forecast"
    ]

    all_cols = [
        "Date",
        "HV_30d",
        #"VDAX",
        "RW_Forecast",
        "EWMA_Forecast",
        "GARCH_Split_Forecast",
        "EGARCH_Split_Forecast",
       # "Roll_GARCH_Forecast",
       # "Roll_EGARCH_Forecast"
    ]

    #col = "Roll_GARCH_Forecast"

    #print("NaN-Anteil:", df_dax[col].isna().mean())
    #print("<= 5*VOL_FLOOR Anteil:", (df_dax[col] <= 5*VOL_FLOOR).mean())

    # Zeig die auffälligen Tage
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
    

    df_plot = plot_volatility_forecasts(
        df=df,
        forecast_specs=[
            ("RW_Forecast", "RW"),
            ("EWMA_Forecast", "EWMA"),
            ("GARCH_Split_Forecast", "Split GARCH"),
            ("EGARCH_Split_Forecast", "Split EGARCH"),
        ],
        asset_name="USD/EUR",
        model_name="Benchmark Models",
        horizon=1,
        split_date="2018-01-01",
        oos_only=True,
        save_path="plots/usdeur_benchmark.png",
        show=True
    )

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
