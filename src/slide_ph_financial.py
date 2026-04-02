"""
Slide 25 — Python: calculando HP de datos financieros
Persistent homology on S&P 500 log-returns via sliding window embedding.

Install: pip install ripser persim yfinance matplotlib numpy
"""
import numpy as np
import yfinance as yf
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt


def sliding_window_cloud(series, window=50, d=3, tau=1):
    """Build point cloud from a window of a time series."""
    N = len(series)
    points = []
    for t in range(N - (d - 1) * tau):
        pt = [series[t + k * tau] for k in range(d)]
        points.append(pt)
    return np.array(points)


if __name__ == "__main__":
    # --- 1. Download S&P 500 data ---
    sp500 = yf.download("^GSPC", start="2005-01-01", end="2010-12-31")
    log_returns = np.log(sp500['Close']).diff().dropna().values.squeeze()

    # --- 2. Compute persistent homology on one window ---
    W = 50
    cloud = sliding_window_cloud(log_returns[:W], d=3, tau=1)

    result = ripser(cloud, maxdim=1)   # compute H0 and H1
    diagrams = result['dgms']          # diagrams[0]=H0, diagrams[1]=H1

    # --- 3. Visualize ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_diagrams(diagrams, ax=axes[0], title="Persistence Diagram")

    for (b, d_val) in diagrams[1]:
        if d_val == np.inf:
            continue
        axes[1].plot([b, d_val], [0, 0], 'r-', linewidth=2, alpha=0.7)
    axes[1].set_title("H₁ Barcode")

    plt.tight_layout()
    plt.savefig("sp500_tda.png", dpi=150)
    print("Saved sp500_tda.png")
