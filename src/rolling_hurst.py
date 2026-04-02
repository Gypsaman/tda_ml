"""
Rolling Hurst Exponent Estimator
=================================
Estimates the Hurst exponent H over a rolling window on financial return series.

H < 0.5  → mean-reverting (anti-persistent)
H = 0.5  → random walk (no memory)
H > 0.5  → trending (persistent / long-memory)

Two methods implemented:
  - R/S  : classic Hurst (1951) rescaled range analysis
  - DFA  : detrended fluctuation analysis (Peng et al. 1994)

Usage
-----
    python rolling_hurst.py                     # S&P 500, last 5 years, RS method
    python rolling_hurst.py --ticker AAPL --method dfa --window 126
    python rolling_hurst.py --ticker BTC-USD --period 10y --window 252
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------

def hurst_rs(ts: np.ndarray) -> float:
    """
    Estimate H via R/S (rescaled range) analysis.

    Fit  log E[R/S(n)] ~ H * log(n)  over multiple sub-window sizes.

    Parameters
    ----------
    ts : 1-D array of log-returns (or prices).

    Returns
    -------
    H : float in [0, 1], or np.nan if estimation fails.
    """
    n = len(ts)
    if n < 20:
        return np.nan

    lags = np.unique(np.floor(np.logspace(np.log10(10), np.log10(n // 2), 20)).astype(int))
    lags = lags[lags >= 10]

    rs_values = []
    valid_lags = []

    for lag in lags:
        sub_rs = []
        for start in range(0, n - lag + 1, lag):
            chunk = ts[start : start + lag].copy()
            chunk -= chunk.mean()
            std = chunk.std(ddof=1)
            if std == 0:
                continue
            cumdev = np.cumsum(chunk)
            R = cumdev.max() - cumdev.min()
            sub_rs.append(R / std)
        if sub_rs:
            rs_values.append(np.mean(sub_rs))
            valid_lags.append(lag)

    if len(valid_lags) < 4:
        return np.nan

    log_lags = np.log(valid_lags)
    log_rs   = np.log(rs_values)
    # Linear regression: log(R/S) = H * log(n) + const
    H, _ = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(H, 0.0, 1.0))


def hurst_dfa(ts: np.ndarray) -> float:
    """
    Estimate H via Detrended Fluctuation Analysis (DFA).

    Fit  log F(n) ~ H * log(n)  over multiple segment sizes.

    Parameters
    ----------
    ts : 1-D array of log-returns.

    Returns
    -------
    H : float in [0, 1], or np.nan if estimation fails.
    """
    n = len(ts)
    if n < 20:
        return np.nan

    # Integrated series (profile)
    profile = np.cumsum(ts - ts.mean())

    scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(n // 4), 20)).astype(int))
    scales = scales[scales >= 10]

    fluctuations = []
    valid_scales = []

    for scale in scales:
        n_segs = n // scale
        if n_segs < 2:
            continue
        rms_list = []
        for k in range(n_segs):
            seg = profile[k * scale : (k + 1) * scale]
            x   = np.arange(scale)
            # Remove linear trend
            coeffs = np.polyfit(x, seg, 1)
            trend  = np.polyval(coeffs, x)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluctuations.append(np.mean(rms_list))
        valid_scales.append(scale)

    if len(valid_scales) < 4:
        return np.nan

    log_scales = np.log(valid_scales)
    log_fluct  = np.log(fluctuations)
    H, _ = np.polyfit(log_scales, log_fluct, 1)
    return float(np.clip(H, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Rolling estimator
# ---------------------------------------------------------------------------

def rolling_hurst(
    returns: pd.Series,
    window: int = 252,
    method: str = "rs",
    step: int = 1,
) -> pd.Series:
    """
    Compute a rolling Hurst exponent over `returns`.

    Parameters
    ----------
    returns : pd.Series of log-returns, indexed by date.
    window  : look-back window in observations (default 252 ≈ 1 trading year).
    method  : 'rs' (R/S analysis) or 'dfa'.
    step    : compute every `step` observations (1 = daily, set higher to speed up).

    Returns
    -------
    pd.Series of H estimates aligned to the right edge of each window.
    """
    estimator = hurst_rs if method == "rs" else hurst_dfa
    arr = returns.to_numpy()
    n   = len(arr)

    indices = range(window - 1, n, step)
    h_vals  = [estimator(arr[i - window + 1 : i + 1]) for i in indices]
    dates   = [returns.index[i] for i in indices]

    return pd.Series(h_vals, index=dates, name=f"H_{method.upper()}")


# ---------------------------------------------------------------------------
# Data fetch + pipeline
# ---------------------------------------------------------------------------

def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    print(f"Downloading {ticker} ({period}) …")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    prices: pd.Series,
    returns: pd.Series,
    h_series: pd.Series,
    ticker: str,
    window: int,
    method: str,
    save_path: str | None = None,
):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    fig.suptitle(
        f"{ticker} — Rolling Hurst Exponent (window={window}d, method={method.upper()})",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: price
    ax = axes[0]
    ax.plot(prices.index, prices.values, color="steelblue", linewidth=0.8)
    ax.set_ylabel("Price (USD)")
    ax.set_title("Adjusted Close")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Panel 2: log-returns
    ax = axes[1]
    ax.plot(returns.index, returns.values, color="gray", linewidth=0.5, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Log return")
    ax.set_title("Daily Log Returns")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Panel 3: rolling Hurst
    ax = axes[2]
    h_vals = h_series.values
    dates  = h_series.index

    # Colour by regime
    ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--", label="H = 0.5 (random walk)")
    ax.axhspan(0.0, 0.5, alpha=0.06, color="blue",  label="Mean-reverting (H < 0.5)")
    ax.axhspan(0.5, 1.0, alpha=0.06, color="red",   label="Persistent (H > 0.5)")

    # Scatter coloured by value
    sc = ax.scatter(dates, h_vals, c=h_vals, cmap="RdYlGn", vmin=0.3, vmax=0.7,
                    s=4, zorder=3)
    ax.plot(dates, h_vals, color="dimgray", linewidth=0.5, alpha=0.4, zorder=2)

    # Smoothed trend
    if len(h_vals) > 30:
        smooth = pd.Series(h_vals, index=dates).rolling(30, min_periods=1).mean()
        ax.plot(smooth.index, smooth.values, color="navy", linewidth=1.5,
                label="30-obs smoothed", zorder=4)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("H")
    ax.set_title(f"Rolling Hurst Exponent ({method.upper()}, window={window}d)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(sc, ax=ax, label="H value")

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(h_series: pd.Series, ticker: str, method: str) -> None:
    clean = h_series.dropna()
    if clean.empty:
        print("No valid H estimates.")
        return

    pct_persistent   = (clean > 0.5).mean() * 100
    pct_mean_rev     = (clean < 0.5).mean() * 100
    pct_random_walk  = ((clean >= 0.49) & (clean <= 0.51)).mean() * 100

    print(f"\n{'='*55}")
    print(f"  Hurst Summary: {ticker}  [{method.upper()}]")
    print(f"{'='*55}")
    print(f"  Observations   : {len(clean)}")
    print(f"  Mean H         : {clean.mean():.4f}")
    print(f"  Median H       : {clean.median():.4f}")
    print(f"  Std H          : {clean.std():.4f}")
    print(f"  Min / Max      : {clean.min():.4f} / {clean.max():.4f}")
    print(f"  % Persistent   : {pct_persistent:.1f}%  (H > 0.5)")
    print(f"  % Mean-rev     : {pct_mean_rev:.1f}%  (H < 0.5)")
    print(f"  % Near RW      : {pct_random_walk:.1f}%  (0.49 ≤ H ≤ 0.51)")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rolling Hurst exponent estimator on financial data."
    )
    p.add_argument("--ticker", default="^GSPC",
                   help="Yahoo Finance ticker (default: ^GSPC = S&P 500)")
    p.add_argument("--period", default="10y",
                   help="Download period: 1y 5y 10y max (default: 10y)")
    p.add_argument("--window", type=int, default=252,
                   help="Rolling window in trading days (default: 252)")
    p.add_argument("--method", choices=["rs", "dfa"], default="rs",
                   help="Estimation method: rs or dfa (default: rs)")
    p.add_argument("--step",   type=int, default=5,
                   help="Compute every N days (default: 5, set to 1 for full resolution)")
    p.add_argument("--save",   default=None,
                   help="Path to save the plot PNG (default: display interactively)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Fetch
    df = fetch_data(args.ticker, args.period)
    close = df["Close"].squeeze()  # handle MultiIndex columns from yfinance

    # 2. Log returns
    returns = compute_log_returns(close)

    print(f"Price series   : {close.index[0].date()} → {close.index[-1].date()} ({len(close)} obs)")
    print(f"Returns series : {returns.index[0].date()} → {returns.index[-1].date()} ({len(returns)} obs)")
    print(f"Window         : {args.window} days | Method: {args.method.upper()} | Step: {args.step}")

    # 3. Rolling Hurst
    print("Computing rolling Hurst exponent …")
    h = rolling_hurst(returns, window=args.window, method=args.method, step=args.step)
    print(f"H estimates    : {h.dropna().shape[0]} points")

    # 4. Summary
    print_summary(h, args.ticker, args.method)

    # 5. Plot
    save = args.save
    if save is None:
        out_dir = Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        ticker_clean = args.ticker.replace("^", "").replace("-", "_")
        save = str(out_dir / f"hurst_{ticker_clean}_{args.method}_{args.window}d.png")
    plot_results(close, returns, h, ticker=args.ticker,
                 window=args.window, method=args.method, save_path=save)


if __name__ == "__main__":
    main()
