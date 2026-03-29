import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.ticker as mtick
from scipy.stats import norm

##################################################################
# Script for plotting volatility forecasts, autocorrelation
# functions, and return distributions
##################################################################


def plot_volatility_forecasts(
    df,
    forecast_specs,
    asset_name,
    model_name,
    horizon,
    hv_col="HV_30d",
    date_col="Date",
    xlim=None,
    split_date=None,
    oos_only=False,
    figsize=(12, 5),
    observed_label="Historical volatility",
    xlabel="Date",
    ylabel="Volatility",
    linewidth_hv=1.8,
    linewidth_fc=1.0,
    save_path=None,
    show=True,
):
    """
    Plot historical volatility together with one or more forecast series.
    """

    title = f"{asset_name}: {model_name} vs Historical Volatility (k={horizon})"

    dfx = df.copy()

    # basic input validation
    if date_col not in dfx.columns:
        raise ValueError(f"Missing date column '{date_col}'.")
    if hv_col not in dfx.columns:
        raise ValueError(f"Missing historical volatility column '{hv_col}'.")

    # parse dates, drop invalid ones and sort chronologically
    dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce")
    dfx = dfx.dropna(subset=[date_col]).sort_values(date_col)

    # restrict to the out-of-sample period if requested
    if oos_only and split_date is not None:
        split_ts = pd.Timestamp(split_date)
        dfx = dfx[dfx[date_col] >= split_ts]

    # rstrict to a manually chosen date window if provided
    if xlim is not None:
        start = pd.Timestamp(xlim[0])
        end = pd.Timestamp(xlim[1])
        dfx = dfx[(dfx[date_col] >= start) & (dfx[date_col] <= end)]

    # convert the observed volatility series and all forecast series to numeric
    # replace infinities with NaN to avoid plotting issues
    cols_to_clean = [hv_col] + [col for col, _ in forecast_specs if col in dfx.columns]
    for col in cols_to_clean:
        dfx[col] = pd.to_numeric(dfx[col], errors="coerce")
        dfx[col] = dfx[col].replace([np.inf, -np.inf], np.nan)

    # fixed colors for common model labels to ensure consistent appearance
    color_map = {
        "RW": "#6C757D",
        "EWMA": "#1F77B4",
        "GARCH": "#D62728",
        "GARCH (Split)": "#D62728",
        "Split GARCH": "#D62728",
        "Roll GARCH": "#D62728",
        "EGARCH": "#2CA02C",
        "EGARCH (Split)": "#2CA02C",
        "Split EGARCH": "#2CA02C",
        "Roll EGARCH": "#2CA02C",
        "MLP": "#9467BD",
        "LSTM": "#FF7F0E",
        "G-LSTM": "#8C564B",
        "E-LSTM": "#17BECF",
    }

    # fallback colors for models not covered by the fixed color map
    fallback_colors = [
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
        "#9467BD", "#8C564B", "#17BECF", "#BCBD22"
    ]
    fallback_idx = 0

    def get_color(col, label):
        """
        Return a plotting color for a forecast series.

        Priority:
        1. exact match on legend label
        2. exact match on column name
        3. next fallback color
        """
        nonlocal fallback_idx
        if label in color_map:
            return color_map[label]
        if col in color_map:
            return color_map[col]
        color = fallback_colors[fallback_idx % len(fallback_colors)]
        fallback_idx += 1
        return color

    fig, ax = plt.subplots(figsize=figsize)

    # plot observed historical volatility as the main reference series
    ax.plot(
        dfx[date_col],
        dfx[hv_col],
        label=observed_label,
        color="#111111",
        linewidth=linewidth_hv,
        solid_capstyle="round",
        zorder=3,
    )

    # plot all forecast series listed in forecast_specs
    plotted_forecasts = 0
    for col, label in forecast_specs:
        if col not in dfx.columns:
            continue

        ax.plot(
            dfx[date_col],
            dfx[col],
            label=label,
            color=get_color(col, label),
            linestyle="-",
            linewidth=linewidth_fc,
            alpha=0.95,
            zorder=2,
        )
        plotted_forecasts += 1

    # stop if none of the requested forecast columns are present
    if plotted_forecasts == 0:
        raise ValueError("No forecast columns from forecast_specs were found in df.")

    # draw a vertical split line if the split date lies inside the plotted range
    if split_date is not None:
        split_ts = pd.Timestamp(split_date)
        if len(dfx) > 0 and dfx[date_col].min() <= split_ts <= dfx[date_col].max():
            ax.axvline(
                split_ts,
                color="#7A7A7A",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                zorder=1,
            )

    # titles and axis labels
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    # visual styling
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(x=0.01)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    # use concise date formatting on the x-axis
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # remove duplicate legend labels while preserving the last handle per label
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="best",
        frameon=True,
        framealpha=0.95,
        edgecolor="#D9D9D9",
        fontsize=10,
        ncol=1,
    )

    fig.tight_layout()

    # save figure if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # show or close figure
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_log_returns(
    df,
    asset_name,
    date_col="Date",
    logreturn_col="LogReturn",
    xlabel="Date",
    ylabel="Log Return",
    figsize=(12, 5),
    linewidth=1.0,
    grid=True,
    save_path=None,
    show=True,
):
    """
    Plot log returns over time.
    """
    title = f"{asset_name}: Log Returns over Time"

    plt.figure(figsize=figsize)
    plt.plot(df[date_col], df[logreturn_col], linewidth=linewidth)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if grid:
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def acf_plot(
    df,
    return_col="LogReturn",
    assetname="Asset",
    lags=50,
    ylim_returns=(-0.10, 0.35),
    ylim_squared=(-0.05, 0.35),
    save_path_r=None,
    save_path_r2=None,
    show=True
):
    """
    Plot the autocorrelation function of log returns and squared log returns.
    """
    df = df.copy()

    # raw return series and volatility-proxy series
    r = df[return_col].dropna()
    r2 = r**2

    # ----------------------------
    # ACF of log returns
    # ----------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    plot_acf(
        r,
        lags=lags,
        ax=ax1,
        zero=False,
        alpha=0.05,
        title=f"{assetname}: Log Return Autocorrelation Function"
    )

    ax1.set_ylim(ylim_returns)
    ax1.set_xlabel("Lag", fontsize=11)
    ax1.set_ylabel("ACF", fontsize=11)
    ax1.set_title(f"{assetname}: Log Return Autocorrelation Function", fontsize=13, pad=12)

    ax1.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="both", labelsize=10)

    fig1.tight_layout()

    if save_path_r is not None:
        save_path_r = Path(save_path_r)
        save_path_r.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(save_path_r, dpi=300, bbox_inches="tight")

    # ----------------------------
    # ACF of squared log returns
    # ----------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    plot_acf(
        r2,
        lags=lags,
        ax=ax2,
        zero=False,
        alpha=0.05,
        title=f"{assetname}: Squared Log Return Autocorrelation Function"
    )

    ax2.set_ylim(ylim_squared)
    ax2.set_xlabel("Lag", fontsize=11)
    ax2.set_ylabel("ACF", fontsize=11)
    ax2.set_title(f"{assetname}: Squared Log Return Autocorrelation Function", fontsize=13, pad=12)

    ax2.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", labelsize=10)

    fig2.tight_layout()

    if save_path_r2 is not None:
        save_path_r2 = Path(save_path_r2)
        save_path_r2.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(save_path_r2, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    return fig1, fig2


def plot_distribution(
    df,
    asset_name,
    return_col="LogReturn",
    bins=80,
    figsize=(9, 5),
    xlabel="Log return",
    ylabel="Density",
    save_path=None,
    show=True,
):
    """
    Plot the empirical distribution of log returns and overlay a fitted
    normal density.
    """
    # remove missing values from the return series
    r = df[return_col].dropna()

    # estimate mean and standard deviation for the fitted normal distribution
    mu = r.mean()
    sigma = r.std(ddof=0)

    title = f"{asset_name}: Distribution of Log Returns"

    fig, ax = plt.subplots(figsize=figsize)

    # empirical histogram, scaled to density
    ax.hist(
        r,
        bins=bins,
        density=True,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.95,
        label="Empirical distribution",
        zorder=2,
    )

    # fitted normal density over the same support
    x = np.linspace(r.min(), r.max(), 500)
    pdf = norm.pdf(x, loc=mu, scale=sigma)

    ax.plot(
        x,
        pdf,
        color="orange",
        linestyle="-",
        linewidth=1.2,
        label="Normal distribution",
        zorder=3,
    )

    # mark the sample mean with a vertical line
    ax.axvline(
        mu,
        linewidth=1.2,
        color="black",
        linestyle=":",
        label="Mean",
        zorder=4,
    )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    # format x-axis as percentages because returns are stored in decimal form
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))

    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.95,
        edgecolor="#D9D9D9",
        fontsize=10,
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax