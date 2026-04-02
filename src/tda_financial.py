"""
TDA Financial Time Series Analysis
===================================

Performs persistent homology analysis on sliding-window embeddings of
financial time series, implementing the topological Hurst estimator from:

  "Concentration Inequalities for Persistent Homology of Long-Range Dependent
   Point Clouds, with a Topological Hurst Estimator"

WHY VIX?
--------
The CBOE Volatility Index is the ideal test case:
  - Exhibits strong long-memory (H ≈ 0.65–0.85 empirically)
  - Volatility clustering creates loops in the delay embedding: the market
    cycles through fear/calm states, which persistent H1 captures as 1-cycles
  - Dramatic contrast between crisis regimes (2008, 2020) and calm periods
  - The paper explicitly studies VIX; this code reproduces Section 6 numerics

Pipeline
--------
  1. Simulate fBm (Davies-Harte) with known H → calibrate α(H)
  2. Download VIX and S&P 500 data via yfinance
  3. Build sliding-window embeddings  P_{d,τ}^{(n)}  (d=3, τ=1 default)
  4. Compute persistent H1 via Ripser (or GUDHI fallback)
  5. Compute barcode count function  L_n^(1)(ε)  and fit log-log slope
  6. Estimate  Ĥ_top  by inverting calibrated α(H)
  7. Rolling analysis: Ĥ_top vs Ĥ_RS, detect regime changes
  8. Persistence diagram snapshots at 2008 crisis / 2020 COVID / calm period

Usage
-----
    python tda_financial.py                    # full analysis, auto-save
    python tda_financial.py --no-save          # interactive display
    python tda_financial.py --quick            # skip fBm calibration
    python tda_financial.py --ticker ^VIX --period 20y
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from scipy.stats import linregress
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Persistent homology backend  (ripser preferred, gudhi fallback)
# ---------------------------------------------------------------------------

try:
    from ripser import ripser as _ripser

    def compute_ph(points: np.ndarray, max_dim: int = 1) -> list[np.ndarray]:
        """Return list of persistence diagrams dgms[k] for k = 0 … max_dim."""
        result = _ripser(points, maxdim=max_dim)
        return result["dgms"]

    PH_BACKEND = "ripser"

except ImportError:
    try:
        import gudhi

        def compute_ph(points: np.ndarray, max_dim: int = 1) -> list[np.ndarray]:
            rips = gudhi.RipsComplex(points=points.tolist())
            st = rips.create_simplex_tree(max_dimension=max_dim + 1)
            st.compute_persistence()
            dgms = []
            for k in range(max_dim + 1):
                pairs = st.persistence_intervals_in_dimension(k)
                arr = np.array(pairs) if len(pairs) > 0 else np.zeros((0, 2))
                # cap infinite bars at finite max (for H0 essential class)
                finite_max = arr[np.isfinite(arr)].max() if np.isfinite(arr).any() else 1.0
                arr[arr == np.inf] = finite_max * 1.5
                dgms.append(arr)
            return dgms

        PH_BACKEND = "gudhi"

    except ImportError:
        raise ImportError(
            "No PH backend found. Install with:\n"
            "  pip install ripser          (recommended)\n"
            "  pip install gudhi           (alternative)"
        )


# ---------------------------------------------------------------------------
# Fractional Brownian Motion simulation  (Davies-Harte / circulant embedding)
# ---------------------------------------------------------------------------

def simulate_fbm(n: int, H: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Simulate n steps of fBm increments (fGn) with Hurst exponent H.

    Uses the Davies-Harte exact method via circulant embedding:
      γ(k) = ½ (|k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H})

    Parameters
    ----------
    n   : number of increments
    H   : Hurst exponent in (0, 1)
    rng : numpy random generator (for reproducibility)

    Returns
    -------
    fbm : cumulative sum of increments, shape (n+1,), starting at 0
    """
    if rng is None:
        rng = np.random.default_rng()

    # Autocovariance of fractional Gaussian noise
    k = np.arange(n)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    gamma[0] = 1.0  # variance = 1

    # Circulant embedding
    row = np.concatenate([gamma, gamma[-1:0:-1]])
    m = len(row)
    eigenvalues = np.fft.fft(row).real
    # Ensure non-negative (numerical noise can produce tiny negatives)
    eigenvalues = np.maximum(eigenvalues, 0)

    # Generate fGn via IFFT
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    w = np.fft.ifft(np.sqrt(eigenvalues) * z).real[:n]
    w = w / w.std()  # unit variance fGn

    return np.concatenate([[0.0], np.cumsum(w)])


# ---------------------------------------------------------------------------
# Time-delay (sliding window) embedding
# ---------------------------------------------------------------------------

def sliding_window_embedding(ts: np.ndarray, d: int = 3, tau: int = 1) -> np.ndarray:
    """
    Build the sliding-window point cloud  P_{d,τ}^{(n)}.

    Parameters
    ----------
    ts  : 1-D time series of length N
    d   : embedding dimension
    tau : lag (time step between coordinates)

    Returns
    -------
    cloud : array of shape (n, d) where n = N - (d-1)*tau
    """
    N = len(ts)
    n = N - (d - 1) * tau
    if n <= 0:
        raise ValueError(f"Series too short for d={d}, tau={tau}. Need N > {(d-1)*tau}.")
    return np.column_stack([ts[i * tau : i * tau + n] for i in range(d)])


# ---------------------------------------------------------------------------
# Barcode analysis
# ---------------------------------------------------------------------------

def barcode_count(dgm: np.ndarray, eps_grid: np.ndarray) -> np.ndarray:
    """
    Compute L_n^(1)(ε) = #{bars in dgm with lifetime > ε} for each ε.

    Parameters
    ----------
    dgm      : H1 persistence diagram, shape (m, 2)
    eps_grid : array of ε values

    Returns
    -------
    counts : array of shape (len(eps_grid),)
    """
    if len(dgm) == 0:
        return np.zeros(len(eps_grid))
    lifetimes = dgm[:, 1] - dgm[:, 0]
    # Remove infinite bars (essential classes; ripser represents as inf)
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    return np.array([(lifetimes > eps).sum() for eps in eps_grid], dtype=float)


def fit_alpha(eps_grid: np.ndarray, counts: np.ndarray) -> float:
    """
    Fit log L = -α log ε + const via OLS.  Returns α̂ (the slope magnitude).

    Only uses ε values where counts > 0 and counts < max (avoid boundary effects).
    """
    mask = (counts > 0) & (counts < counts.max())
    if mask.sum() < 3:
        return np.nan
    log_eps = np.log(eps_grid[mask])
    log_L   = np.log(counts[mask])
    slope, _, r, _, _ = linregress(log_eps, log_L)
    return float(-slope)  # α̂ = -slope


def total_persistence(dgm: np.ndarray) -> float:
    """Sum of bar lifetimes in the H1 diagram (area under Betti curve)."""
    if len(dgm) == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    return float(np.sum(lifetimes[np.isfinite(lifetimes)]))


# ---------------------------------------------------------------------------
# Calibration of α(H) via fBm simulations
# ---------------------------------------------------------------------------

def calibrate_alpha(
    H_values: np.ndarray,
    n: int = 1000,
    d: int = 3,
    tau: int = 1,
    n_trials: int = 10,
    eps_grid: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate α(H) for each H in H_values by Monte Carlo fBm simulation.

    Returns
    -------
    H_values : input array (unchanged)
    alphas   : estimated α(H) for each H
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if eps_grid is None:
        eps_grid = np.logspace(-2.0, 0.5, 40)

    alphas = []
    for H in H_values:
        trial_alphas = []
        for _ in range(n_trials):
            fbm = simulate_fbm(n, H, rng=rng)
            cloud = sliding_window_embedding(fbm, d=d, tau=tau)
            # Normalize cloud to unit scale for comparability
            scale = cloud.std()
            if scale == 0:
                continue
            cloud = cloud / scale
            try:
                dgms = compute_ph(cloud, max_dim=1)
                h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
                counts = barcode_count(h1, eps_grid)
                a = fit_alpha(eps_grid, counts)
                if np.isfinite(a):
                    trial_alphas.append(a)
            except Exception:
                continue
        if trial_alphas:
            alphas.append(np.mean(trial_alphas))
            if verbose:
                print(f"  H={H:.2f} → α̂ = {alphas[-1]:.3f}  (±{np.std(trial_alphas):.3f}, {len(trial_alphas)} trials)")
        else:
            alphas.append(np.nan)
            if verbose:
                print(f"  H={H:.2f} → α̂ = NaN (all trials failed)")

    return H_values, np.array(alphas)


def make_alpha_inverter(H_values: np.ndarray, alphas: np.ndarray):
    """Return a function α → H by linear interpolation on the calibrated table."""
    valid = np.isfinite(alphas)
    H_v = H_values[valid]
    a_v = alphas[valid]
    # Sort by α for interpolation
    order = np.argsort(a_v)
    H_sorted = H_v[order]
    a_sorted = a_v[order]

    def invert(alpha_hat: float) -> float:
        return float(np.interp(alpha_hat, a_sorted, H_sorted,
                               left=H_sorted[0], right=H_sorted[-1]))

    return invert


# ---------------------------------------------------------------------------
# Topological Hurst estimator on a time series window
# ---------------------------------------------------------------------------

def topo_hurst(
    ts: np.ndarray,
    alpha_inverter,
    d: int = 3,
    tau: int = 1,
    eps_grid: np.ndarray | None = None,
) -> float:
    """
    Estimate H_top for a 1-D time series window.

    Steps:
      1. Build sliding-window embedding (normalised)
      2. Compute H1 persistence diagram
      3. Fit α̂ from log L_n^(1)(ε) vs log ε
      4. Return α_inverter(α̂)
    """
    if eps_grid is None:
        eps_grid = np.logspace(-2.0, 0.5, 40)

    cloud = sliding_window_embedding(ts, d=d, tau=tau)
    scale = cloud.std()
    if scale == 0:
        return np.nan
    cloud = cloud / scale

    try:
        dgms = compute_ph(cloud, max_dim=1)
        h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
        counts = barcode_count(h1, eps_grid)
        alpha_hat = fit_alpha(eps_grid, counts)
        if not np.isfinite(alpha_hat):
            return np.nan
        return alpha_inverter(alpha_hat)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Classical R/S Hurst (self-contained copy from rolling_hurst.py)
# ---------------------------------------------------------------------------

def hurst_rs(ts: np.ndarray) -> float:
    n = len(ts)
    if n < 20:
        return np.nan
    lags = np.unique(np.floor(np.logspace(np.log10(10), np.log10(n // 2), 20)).astype(int))
    lags = lags[lags >= 10]
    rs_values, valid_lags = [], []
    for lag in lags:
        sub_rs = []
        for start in range(0, n - lag + 1, lag):
            chunk = ts[start : start + lag].copy() - ts[start : start + lag].mean()
            std = chunk.std(ddof=1)
            if std == 0:
                continue
            cumdev = np.cumsum(chunk)
            sub_rs.append((cumdev.max() - cumdev.min()) / std)
        if sub_rs:
            rs_values.append(np.mean(sub_rs))
            valid_lags.append(lag)
    if len(valid_lags) < 4:
        return np.nan
    H, _ = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
    return float(np.clip(H, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_series(ticker: str, period: str) -> pd.Series:
    print(f"  Downloading {ticker} ({period}) …", end=" ", flush=True)
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for '{ticker}'.")
    close = df["Close"].squeeze()
    print(f"{len(close)} observations ({close.index[0].date()} – {close.index[-1].date()})")
    return close


# ---------------------------------------------------------------------------
# Rolling analysis
# ---------------------------------------------------------------------------

def rolling_tda(
    ts: np.ndarray,
    dates: pd.DatetimeIndex,
    window: int,
    alpha_inverter,
    step: int = 5,
    d: int = 3,
    tau: int = 1,
    eps_grid: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute rolling H_top, H_RS, and total H1 persistence over a time series.

    Returns three pd.Series indexed by the right-edge date of each window:
      h_top   : topological Hurst estimate
      h_rs    : R/S Hurst estimate
      tp_h1   : total H1 persistence (sum of bar lifetimes) — scale-normalised
    """
    n = len(ts)
    indices = range(window - 1, n, step)
    h_top_vals, h_rs_vals, tp_vals, roll_dates = [], [], [], []

    total = len(list(indices))
    for i, idx in enumerate(indices):
        if verbose and i % max(1, total // 20) == 0:
            pct = 100 * i / total
            print(f"    {pct:5.1f}%  [{dates[idx].date()}]", flush=True)
        chunk = ts[idx - window + 1 : idx + 1]
        h_top_vals.append(topo_hurst(chunk, alpha_inverter, d=d, tau=tau, eps_grid=eps_grid))
        h_rs_vals.append(hurst_rs(chunk))
        # Total persistence: compute directly (no calibration needed)
        try:
            cloud = sliding_window_embedding(chunk, d=d, tau=tau)
            scale = cloud.std()
            cloud = cloud / scale if scale > 0 else cloud
            dgms = compute_ph(cloud, max_dim=1)
            h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
            tp_vals.append(total_persistence(h1))
        except Exception:
            tp_vals.append(np.nan)
        roll_dates.append(dates[idx])

    idx_dates = pd.DatetimeIndex(roll_dates)
    h_top = pd.Series(h_top_vals, index=idx_dates, name="H_top")
    h_rs  = pd.Series(h_rs_vals,  index=idx_dates, name="H_RS")
    tp_h1 = pd.Series(tp_vals,    index=idx_dates, name="TotalPers_H1")
    return h_top, h_rs, tp_h1


# ---------------------------------------------------------------------------
# Snapshot persistence diagram for a date range
# ---------------------------------------------------------------------------

def compute_snapshot(
    ts: np.ndarray,
    dates: pd.DatetimeIndex,
    start: str,
    end: str,
    d: int = 3,
    tau: int = 1,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (H0 diagram, H1 diagram, label) for the time window [start, end]."""
    mask = (dates >= start) & (dates <= end)
    chunk = ts[mask]
    cloud = sliding_window_embedding(chunk, d=d, tau=tau)
    scale = cloud.std()
    cloud = cloud / scale if scale > 0 else cloud
    dgms = compute_ph(cloud, max_dim=1)
    h0 = dgms[0] if len(dgms) > 0 else np.zeros((0, 2))
    h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
    label = f"{start[:7]} – {end[:7]}"
    return h0, h1, label


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_calibration(H_values, alphas, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = np.isfinite(alphas)
    ax.plot(H_values[valid], alphas[valid], "o-", color="steelblue", lw=2, ms=7)
    ax.set_xlabel("Hurst exponent $H$", fontsize=12)
    ax.set_ylabel(r"Estimated $\hat{\alpha}(H)$", fontsize=12)
    ax.set_title(r"Calibration: $\alpha(H)$ from fBm simulations", fontsize=13)
    ax.grid(alpha=0.3)
    # Annotate monotonicity
    ax.text(0.05, 0.93, r"$\alpha(H)$ strictly increasing in $H$ (Theorem 2)",
            transform=ax.transAxes, fontsize=9, color="dimgray",
            va="top", style="italic")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_persistence_snapshots(snapshots: list[tuple], title: str, save_path=None):
    """
    Plot persistence diagrams for multiple time-window snapshots side-by-side.
    snapshots : list of (h0, h1, label) tuples
    """
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (h0, h1, label) in zip(axes, snapshots):
        # Determine plot limits
        all_finite = []
        for dgm in [h0, h1]:
            if len(dgm) > 0:
                finite_vals = dgm[np.isfinite(dgm)]
                if len(finite_vals):
                    all_finite.extend(finite_vals.tolist())
        if all_finite:
            vmax = max(all_finite) * 1.1
        else:
            vmax = 1.0

        ax.plot([0, vmax], [0, vmax], "k--", lw=0.8, alpha=0.5, label="diagonal")
        if len(h0) > 0:
            finite_h0 = h0[np.isfinite(h0).all(axis=1)]
            ax.scatter(finite_h0[:, 0], finite_h0[:, 1],
                       s=30, c="steelblue", alpha=0.7, label=f"H₀ ({len(h0)})")
        if len(h1) > 0:
            finite_h1 = h1[np.isfinite(h1).all(axis=1)]
            ax.scatter(finite_h1[:, 0], finite_h1[:, 1],
                       s=40, c="crimson", marker="D", alpha=0.7,
                       label=f"H₁ ({len(h1)})")
        ax.set_xlim(-0.02, vmax)
        ax.set_ylim(-0.02, vmax)
        ax.set_xlabel("Birth", fontsize=10)
        ax.set_ylabel("Death", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)

        # Annotate total persistence
        tp_h1 = sum(
            (d - b) for b, d in h1 if np.isfinite(d)
        ) if len(h1) > 0 else 0.0
        ax.text(0.04, 0.96, f"Total $H_1$ pers.: {tp_h1:.3f}",
                transform=ax.transAxes, fontsize=8, va="top", color="crimson")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_rolling_comparison(
    prices: pd.Series,
    h_top: pd.Series,
    h_rs: pd.Series,
    tp_h1: pd.Series,
    ticker: str,
    window: int,
    save_path=None,
):
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 1, hspace=0.40)

    # Panel 1: price/level
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(prices.index, prices.values, color="steelblue", lw=0.8)
    ax0.set_ylabel("Level", fontsize=10)
    ax0.set_title(f"{ticker} — Price / Level", fontsize=11)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax0.xaxis.set_major_locator(mdates.YearLocator())
    ax0.grid(alpha=0.2)

    # Panel 2: total H1 persistence (topological complexity)
    ax1 = fig.add_subplot(gs[1])
    clean_tp = tp_h1.dropna()
    ax1.fill_between(clean_tp.index, clean_tp.values, alpha=0.3, color="purple")
    ax1.plot(clean_tp.index, clean_tp.values, color="purple", lw=0.8, alpha=0.7)
    smooth_tp = clean_tp.rolling(10, min_periods=1).mean()
    ax1.plot(smooth_tp.index, smooth_tp.values, color="indigo", lw=2.0,
             label="10-obs MA")
    ax1.set_ylabel("Total $H_1$ pers.", fontsize=10)
    ax1.set_title(
        f"Rolling Total H₁ Persistence  (topological complexity, window={window}d)\n"
        "Spikes = more 1-cycles in delay embedding = looping/oscillatory regimes",
        fontsize=10,
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    # Panel 3: H_top
    ax2 = fig.add_subplot(gs[2])
    clean_top = h_top.dropna()
    ax2.axhline(0.5, color="black", lw=1.0, ls="--", alpha=0.6, label="H = 0.5 (random walk)")
    ax2.axhspan(0.5, 1.0, alpha=0.05, color="red")
    ax2.axhspan(0.0, 0.5, alpha=0.05, color="blue")
    ax2.plot(clean_top.index, clean_top.values, color="crimson", lw=1.0, alpha=0.6)
    smooth_top = clean_top.rolling(10, min_periods=1).mean()
    ax2.plot(smooth_top.index, smooth_top.values, color="darkred", lw=2.0,
             label="$\\hat{H}_{\\mathrm{top}}$ (10-obs MA)")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("$\\hat{H}_{\\mathrm{top}}$", fontsize=10)
    ax2.set_title(f"Topological Hurst Estimator (window={window}d, fBm-calibrated)", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    # Panel 4: H_top vs H_RS overlay
    ax3 = fig.add_subplot(gs[3])
    clean_rs = h_rs.dropna()
    ax3.axhline(0.5, color="black", lw=1.0, ls="--", alpha=0.6)
    ax3.plot(clean_rs.index, clean_rs.values, color="steelblue", lw=1.2,
             alpha=0.8, label="$\\hat{H}_{\\mathrm{RS}}$  (R/S classical)")
    ax3.plot(smooth_top.index, smooth_top.values, color="darkred", lw=1.5,
             alpha=0.9, label="$\\hat{H}_{\\mathrm{top}}$ (topological, smoothed)")
    ax3.set_ylim(0.0, 1.0)
    ax3.set_ylabel("$H$", fontsize=10)
    ax3.set_title("Comparison: $\\hat{H}_{\\mathrm{top}}$ vs $\\hat{H}_{\\mathrm{RS}}$", fontsize=11)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.2)

    fig.suptitle(
        f"{ticker} — TDA Analysis  (d=3, τ=1, window={window}d, backend={PH_BACKEND})",
        fontsize=13, fontweight="bold",
    )
    fig.autofmt_xdate()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_barcode_count(
    ts: np.ndarray,
    label: str,
    d: int = 3,
    tau: int = 1,
    eps_grid: np.ndarray | None = None,
    save_path=None,
):
    """Plot the barcode count function log L_n^(1)(ε) vs log ε."""
    if eps_grid is None:
        eps_grid = np.logspace(-2.0, 0.5, 50)

    cloud = sliding_window_embedding(ts, d=d, tau=tau)
    scale = cloud.std()
    cloud = cloud / scale if scale > 0 else cloud
    dgms = compute_ph(cloud, max_dim=1)
    h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
    counts = barcode_count(h1, eps_grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    valid = counts > 0
    ax.loglog(eps_grid[valid], counts[valid], "o-", color="crimson", ms=4, lw=1.5)
    ax.set_xlabel(r"Scale $\varepsilon$", fontsize=12)
    ax.set_ylabel(r"$L_n^{(1)}(\varepsilon)$", fontsize=12)
    ax.set_title(f"Barcode count function — {label}", fontsize=12)
    ax.grid(alpha=0.25, which="both")

    # Fit and annotate α̂
    alpha_hat = fit_alpha(eps_grid, counts)
    if np.isfinite(alpha_hat):
        # Draw fitted line
        mask = (counts > 0) & (counts < counts.max())
        x_fit = eps_grid[mask]
        y_fit = counts[mask]
        c = np.exp(np.mean(np.log(y_fit) + alpha_hat * np.log(x_fit)))
        ax.loglog(x_fit, c * x_fit ** (-alpha_hat), "k--", lw=1.2,
                  label=rf"Slope $-\hat{{\alpha}} = -{alpha_hat:.2f}$")
        ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(h_top: pd.Series, h_rs: pd.Series, ticker: str) -> None:
    top = h_top.dropna()
    rs  = h_rs.dropna()
    print(f"\n{'='*58}")
    print(f"  TDA Hurst Summary: {ticker}")
    print(f"{'='*58}")
    for name, s in [("H_top", top), ("H_RS ", rs)]:
        if s.empty:
            print(f"  {name}: no valid estimates")
            continue
        print(f"  {name}  mean={s.mean():.3f}  median={s.median():.3f}  "
              f"std={s.std():.3f}  range=[{s.min():.3f}, {s.max():.3f}]  "
              f"n={len(s)}")
    if not top.empty and not rs.empty:
        common = top.dropna().index.intersection(rs.dropna().index)
        if len(common) > 5:
            corr = top.loc[common].corr(rs.loc[common])
            print(f"  Pearson corr(H_top, H_RS) = {corr:.3f}  ({len(common)} common obs)")
    print(f"{'='*58}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TDA persistent homology analysis on financial time series."
    )
    p.add_argument("--ticker",  default="^VIX",
                   help="Yahoo Finance ticker (default: ^VIX)")
    p.add_argument("--period",  default="15y",
                   help="Download period: 5y 10y 15y 20y max (default: 15y)")
    p.add_argument("--window",  type=int, default=252,
                   help="Rolling window in trading days (default: 252 = 1yr)")
    p.add_argument("--step",    type=int, default=10,
                   help="Rolling step in days (default: 10; set to 1 for full res)")
    p.add_argument("--d",       type=int, default=3,
                   help="Embedding dimension (default: 3)")
    p.add_argument("--tau",     type=int, default=1,
                   help="Embedding lag (default: 1)")
    p.add_argument("--quick",   action="store_true",
                   help="Skip fBm calibration (use a rough linear α(H))")
    p.add_argument("--no-save", action="store_true", dest="no_save",
                   help="Display plots interactively instead of saving")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    import os
    src_dir = os.path.dirname(os.path.abspath(__file__))

    def save_or_none(name: str) -> str | None:
        if args.no_save:
            matplotlib.use("TkAgg")
            return None
        return os.path.join(src_dir, name)

    ticker_clean = args.ticker.replace("^", "").replace("-", "_")

    print(f"\n{'='*58}")
    print(f"  TDA Financial Analysis  [{PH_BACKEND} backend]")
    print(f"  Ticker: {args.ticker}   Period: {args.period}")
    print(f"  Embedding: d={args.d}, τ={args.tau}   Window: {args.window}d")
    print(f"{'='*58}")

    eps_grid = np.logspace(-2.0, 0.5, 40)

    # ------------------------------------------------------------------
    # 1. Calibrate α(H) via fBm simulations
    # ------------------------------------------------------------------
    # Calibration n must match the number of embedded points in each rolling window
    cal_n = args.window - (args.d - 1) * args.tau

    if not args.quick:
        print(f"\n[1/5] Calibrating α(H) from fBm simulations (n={cal_n} to match window) …")
        H_cal = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        rng = np.random.default_rng(0)
        H_cal, alphas_cal = calibrate_alpha(
            H_cal, n=cal_n, d=args.d, tau=args.tau,
            n_trials=10, eps_grid=eps_grid, rng=rng, verbose=True,
        )
        plot_calibration(H_cal, alphas_cal,
                         save_path=save_or_none(f"tda_calibration.png"))
        alpha_inv = make_alpha_inverter(H_cal, alphas_cal)
    else:
        print("\n[1/5] Quick mode: using rough linear α(H) = 0.5 + 2.0*(H - 0.5)")
        # Rough linear approximation (replace with calibrated once computed)
        def alpha_inv(a: float) -> float:
            return float(np.clip(0.5 + (a - 1.0) / 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 2. Fetch data
    # ------------------------------------------------------------------
    print(f"\n[2/5] Fetching data …")
    prices = fetch_series(args.ticker, args.period)

    # For VIX: use log-returns; for volatility indices log(VIX) is more stationary
    if "VIX" in args.ticker.upper():
        ts = np.log(prices.values)
        ts_label = f"log({args.ticker})"
    else:
        ts = np.log(prices.values / prices.values[:-1].clip(1e-10))
        ts = np.concatenate([[0.0], ts[1:]])  # pad to align with prices
        ts_label = f"log-returns({args.ticker})"

    dates = prices.index

    # ------------------------------------------------------------------
    # 3. Persistence diagram snapshots (regime analysis)
    # ------------------------------------------------------------------
    print(f"\n[3/5] Computing persistence diagram snapshots …")
    snapshots = []

    snapshot_windows = []
    # Identify available crisis windows
    date_str = [str(d)[:10] for d in dates]
    all_windows = [
        ("2007-06-01", "2008-12-31", "2008 Financial Crisis"),
        ("2019-01-01", "2019-12-31", "2019 Calm"),
        ("2020-02-01", "2020-12-31", "2020 COVID Crash"),
    ]
    for start, end, name in all_windows:
        mask = (dates >= start) & (dates <= end)
        if mask.sum() > (args.d - 1) * args.tau + 10:
            snapshot_windows.append((start, end, name))

    if not snapshot_windows:
        # Fall back to thirds of the available data
        n_total = len(dates)
        thirds = [(0, n_total // 3), (n_total // 3, 2 * n_total // 3),
                  (2 * n_total // 3, n_total)]
        for a, b in thirds:
            snapshot_windows.append((
                str(dates[a])[:10], str(dates[b - 1])[:10], f"{str(dates[a])[:7]}"
            ))

    for start, end, name in snapshot_windows:
        print(f"  Snapshot: {name} ({start} – {end})")
        mask = (dates >= start) & (dates <= end)
        chunk = ts[mask]
        if len(chunk) < 20:
            continue
        cloud = sliding_window_embedding(chunk, d=args.d, tau=args.tau)
        scale = cloud.std()
        cloud = cloud / scale if scale > 0 else cloud
        dgms = compute_ph(cloud, max_dim=1)
        h0 = dgms[0] if len(dgms) > 0 else np.zeros((0, 2))
        h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
        snapshots.append((h0, h1, name))

    if snapshots:
        plot_persistence_snapshots(
            snapshots,
            title=f"{args.ticker} — Persistence Diagrams by Market Regime",
            save_path=save_or_none(f"tda_{ticker_clean}_persistence.png"),
        )

    # ------------------------------------------------------------------
    # 4. Barcode count function for whole series
    # ------------------------------------------------------------------
    print(f"\n[4/5] Computing barcode count function L_n^(1)(ε) …")
    plot_barcode_count(
        ts, label=ts_label, d=args.d, tau=args.tau, eps_grid=eps_grid,
        save_path=save_or_none(f"tda_{ticker_clean}_barcode_count.png"),
    )

    # ------------------------------------------------------------------
    # 5. Rolling TDA: H_top and H_RS
    # ------------------------------------------------------------------
    print(f"\n[5/5] Rolling TDA analysis (window={args.window}d, step={args.step}d) …")
    print(f"  This computes PH for ~{(len(ts) - args.window) // args.step} windows — may take a few minutes.")
    h_top, h_rs, tp_h1 = rolling_tda(
        ts, dates, window=args.window,
        alpha_inverter=alpha_inv,
        step=args.step, d=args.d, tau=args.tau,
        eps_grid=eps_grid, verbose=True,
    )

    print_summary(h_top, h_rs, args.ticker)

    plot_rolling_comparison(
        prices, h_top, h_rs, tp_h1,
        ticker=args.ticker, window=args.window,
        save_path=save_or_none(f"tda_{ticker_clean}_rolling.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
