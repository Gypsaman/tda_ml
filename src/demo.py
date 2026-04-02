"""
TDA for Financial Markets — Demo Script
MCD524: Deep Learning Applied to Finance

Install dependencies:
    pip install ripser persim giotto-tda yfinance numpy scipy matplotlib scikit-learn

Run sections individually or all at once:
    python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Delay embedding — what topology sees in a time series
# ─────────────────────────────────────────────────────────────────────────────

def delay_embed(series, d=2, tau=5):
    """Sliding window (delay) embedding: each window becomes a point in R^d."""
    N = len(series)
    n_pts = N - (d - 1) * tau
    return np.array([[series[t + k * tau] for k in range(d)]
                     for t in range(n_pts)])


def demo_embedding():
    """Show how different signals produce different point cloud geometries."""
    t = np.linspace(0, 4 * np.pi, 300)
    signals = {
        "Periodic (sine)":    np.sin(t) + 0.1 * np.random.randn(len(t)),
        "Random walk":        np.cumsum(0.3 * np.random.randn(len(t))),
        "Trending (H=0.8)":   _fbm_approx(len(t), H=0.8),
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("Delay Embedding: Time Series → Point Cloud Geometry",
                 fontsize=13, fontweight='bold')

    for col, (name, sig) in enumerate(signals.items()):
        # Normalize to [-1, 1]
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        cloud = delay_embed(sig, d=2, tau=10)

        # Top row: time series
        ax = axes[0, col]
        ax.plot(t, sig, color='#4ecdc4', linewidth=0.9, alpha=0.85)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("t"); ax.set_ylabel("x(t)")
        ax.set_facecolor('#111'); ax.tick_params(colors='#aaa')

        # Bottom row: 2D delay embedding
        ax = axes[1, col]
        sc = ax.scatter(cloud[:, 0], cloud[:, 1], c=np.arange(len(cloud)),
                        cmap='plasma', s=3, alpha=0.6)
        ax.set_xlabel("x(t)"); ax.set_ylabel("x(t+τ)")
        ax.set_title(f"Embedding (d=2, τ=10)", fontsize=9)
        ax.set_facecolor('#111'); ax.tick_params(colors='#aaa')

    plt.tight_layout()
    plt.savefig("embedding_demo.png", dpi=150, bbox_inches='tight')
    print("[✓] Saved: embedding_demo.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Persistent homology of financial data
# ─────────────────────────────────────────────────────────────────────────────

def demo_ph_financial():
    """Compute PH of S&P 500 log-returns and plot diagrams + barcodes."""
    try:
        import yfinance as yf
        df = yf.download("^GSPC", start="2005-01-01", end="2010-12-31",
                         auto_adjust=True, progress=False)
        prices = df['Close'].values
        source = "S&P 500 (yfinance)"
    except Exception:
        print("[!] yfinance unavailable — using synthetic data")
        prices = np.cumprod(1 + 0.001 * np.random.randn(1500)) * 1000
        source = "Synthetic (GBM)"

    log_ret = np.diff(np.log(prices))
    print(f"[•] {source}: {len(log_ret)} daily log-returns")

    # Build point cloud from a 50-day window around the 2008 crash
    # Sep 2008 ~ index 940 in 2005–2010 series
    crash_idx = min(940, len(log_ret) - 60)
    window = log_ret[crash_idx - 50 : crash_idx]
    cloud = delay_embed(window, d=3, tau=1)

    try:
        from ripser import ripser
        from persim import plot_diagrams
    except ImportError:
        print("[!] ripser/persim not installed. Run: pip install ripser persim")
        return

    result = ripser(cloud, maxdim=1)
    dgms = result['dgms']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"PH of {source} — 50-day window near crash", fontsize=12)

    plot_diagrams(dgms, ax=axes[0])
    axes[0].set_title("Persistence Diagram (H₀ + H₁)")

    # Barcode
    ax = axes[1]
    h1 = dgms[1]
    h1 = h1[h1[:, 1] < np.inf]
    h0 = dgms[0]
    h0 = h0[h0[:, 1] < np.inf]

    y = 0
    for b, d in sorted(h0, key=lambda x: -(x[1]-x[0])):
        ax.plot([b, d], [y, y], color='#4ecdc4', linewidth=3, alpha=0.8)
        y += 1
    for b, d in sorted(h1, key=lambda x: -(x[1]-x[0])):
        ax.plot([b, d], [y, y], color='#ff6b6b', linewidth=3, alpha=0.9)
        y += 1

    ax.set_xlabel("Filtration value r")
    ax.set_ylabel("Feature index")
    ax.set_title("Barcode (teal=H₀, red=H₁)")
    ax.set_facecolor('#111'); ax.tick_params(colors='#aaa')

    plt.tight_layout()
    plt.savefig("ph_financial.png", dpi=150, bbox_inches='tight')
    print("[✓] Saved: ph_financial.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Sliding window — TDA crash detection
# ─────────────────────────────────────────────────────────────────────────────

def tda_features_window(series_window, d=3, tau=1):
    """Extract 5 TDA features from a time series window."""
    try:
        from ripser import ripser
    except ImportError:
        return np.zeros(5)

    cloud = delay_embed(series_window, d=d, tau=tau)
    if len(cloud) < 5:
        return np.zeros(5)

    h1 = ripser(cloud, maxdim=1)['dgms'][1]
    h1 = h1[h1[:, 1] < np.inf]
    pers = h1[:, 1] - h1[:, 0] if len(h1) > 0 else np.array([0.0])

    return np.array([
        len(pers),              # count of H₁ features
        pers.sum(),             # total persistence (≈ L¹ norm of landscape)
        pers.max(),             # max persistence
        pers.mean(),            # mean persistence
        pers.std(),             # std of persistence
    ])


def demo_crash_detection():
    """Replicate the Gidea & Katz crash detection idea."""
    try:
        import yfinance as yf
        df = yf.download("^GSPC", start="2006-01-01", end="2010-12-31",
                         auto_adjust=True, progress=False)
        prices = df['Close'].values
        dates  = df.index
        source = "S&P 500"
    except Exception:
        print("[!] Using synthetic data (yfinance unavailable)")
        np.random.seed(42)
        # Simple crash simulation
        n = 1000
        prices = np.cumprod(1 + np.concatenate([
            0.0005 + 0.01 * np.random.randn(700),
            0.002  + 0.03 * np.random.randn(100),   # pre-crash vol spike
            -0.005 + 0.04 * np.random.randn(100),   # crash
            0.001  + 0.015 * np.random.randn(100),  # recovery
        ])) * 1000
        dates  = np.arange(n)
        source = "Synthetic"

    log_ret = np.diff(np.log(prices))
    W = 40  # window length (shorter for speed)

    print(f"[•] Computing TDA features on {source} ({len(log_ret)} days)…")
    print("    This may take 1-2 minutes (PH per window)…")

    total_pers = []
    for i in range(W, len(log_ret)):
        window = log_ret[i - W : i]
        feat = tda_features_window(window)
        total_pers.append(feat[1])  # total persistence = L¹ norm

    total_pers = np.array(total_pers)
    price_idx  = prices[W+1:len(total_pers)+W+1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"TDA Crash Detection — {source}", fontsize=13)

    # Price
    x = np.arange(len(price_idx))
    axes[0].plot(x, price_idx, color='#4ecdc4', linewidth=1)
    axes[0].set_ylabel("Price")
    axes[0].set_facecolor('#111'); axes[0].tick_params(colors='#aaa')

    # TDA signal
    # Smooth for visibility
    from numpy.lib.stride_tricks import sliding_window_view
    smooth_len = 10
    if len(total_pers) >= smooth_len:
        smoothed = np.convolve(total_pers, np.ones(smooth_len)/smooth_len, mode='valid')
        x2 = x[smooth_len-1:]
    else:
        smoothed = total_pers
        x2 = x

    axes[1].plot(x2, smoothed, color='#ff6b6b', linewidth=1.2,
                 label='L¹ persistence norm (smoothed)')
    axes[1].set_ylabel("TDA Signal")
    axes[1].set_xlabel("Days")
    axes[1].legend(fontsize=9)
    axes[1].set_facecolor('#111'); axes[1].tick_params(colors='#aaa')

    plt.tight_layout()
    plt.savefig("crash_detection.png", dpi=150, bbox_inches='tight')
    print("[✓] Saved: crash_detection.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: TDA features for ML classification
# ─────────────────────────────────────────────────────────────────────────────

def demo_ml_pipeline():
    """Simple ML pipeline: TDA features → GBM → direction prediction."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        print("[!] scikit-learn not installed. Run: pip install scikit-learn")
        return

    # Use synthetic fBm-like data for reproducibility
    np.random.seed(0)
    N = 600
    H = 0.7
    series = _fbm_approx(N, H=H)
    series += 0.3 * np.random.randn(N)  # add noise

    W = 40
    horizon = 5   # predict 5-day ahead return

    print(f"[•] Building TDA feature matrix (N={N}, W={W})…")

    features, labels = [], []
    for i in range(W, N - horizon):
        window      = series[i - W : i]
        feat        = tda_features_window(window)
        future_ret  = series[i + horizon] - series[i]
        features.append(feat)
        labels.append(1 if future_ret > 0 else 0)

    X = np.array(features)
    y = np.array(labels)

    # Also add simple price-based features for comparison
    price_feats = []
    for i in range(W, N - horizon):
        window = series[i - W : i]
        price_feats.append([
            window[-1] - window[-5],   # 5-day momentum
            window[-1] - window[-10],  # 10-day momentum
            window.std(),              # volatility
            window[-5:].mean() - window[-20:-5].mean(),  # MA crossover
        ])
    X_price = np.array(price_feats)

    split = int(len(X) * 0.7)

    # Evaluate three models
    scaler = StandardScaler()

    models = {
        "Price features only":   X_price,
        "TDA features only":     X,
        "TDA + Price combined":  np.hstack([X, X_price]),
    }

    print("\n  Model comparison:")
    print(f"  {'Model':<30} {'Test Accuracy':>14}")
    print("  " + "-" * 46)
    for name, Xm in models.items():
        Xs = scaler.fit_transform(Xm)
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                         random_state=42)
        clf.fit(Xs[:split], y[:split])
        acc = accuracy_score(y[split:], clf.predict(Xs[split:]))
        print(f"  {name:<30} {acc:>13.1%}")

    print("\n[•] Note: TDA + Price combined typically outperforms either alone.")
    print("    Real financial data shows similar patterns.\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Topological Hurst estimation (simplified demo)
# ─────────────────────────────────────────────────────────────────────────────

def demo_hurst_estimation():
    """
    Demonstrate the topological Hurst estimator idea:
    the log-log slope of L_n(ε) encodes H for fBm.
    """
    try:
        from ripser import ripser
    except ImportError:
        print("[!] ripser not installed")
        return

    H_values  = [0.3, 0.5, 0.6, 0.7, 0.8]
    eps_grid  = np.logspace(-2.0, 0.0, 20)
    n         = 400

    slopes = []
    print("[•] Computing barcode count L_n(ε) for different H values…")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Topological Hurst Estimation", fontsize=12)
    colors = ['#ff6b6b', '#ffd93d', '#4ecdc4', '#a8e6cf', '#e8a0ef']

    for H, col in zip(H_values, colors):
        series = _fbm_approx(n, H=H)
        cloud  = delay_embed(series, d=3, tau=3)
        dgm1   = ripser(cloud, maxdim=1)['dgms'][1]
        h1     = dgm1[dgm1[:, 1] < np.inf]
        pers   = h1[:, 1] - h1[:, 0] if len(h1) > 0 else np.array([1e-6])

        L_eps = np.array([(pers > eps).sum() for eps in eps_grid])
        L_eps = np.maximum(L_eps, 1)  # avoid log(0)

        log_eps  = np.log(eps_grid)
        log_L    = np.log(L_eps)
        # Fit slope in middle range (avoid extremes)
        mid = slice(5, -5)
        slope = np.polyfit(log_eps[mid], log_L[mid], 1)[0]
        slopes.append(slope)

        axes[0].plot(log_eps, log_L, color=col, linewidth=1.8,
                     label=f'H={H:.1f} (slope={slope:.2f})')

    axes[0].set_xlabel("log ε")
    axes[0].set_ylabel("log L_n(ε)")
    axes[0].set_title("Barcode count vs threshold")
    axes[0].legend(fontsize=8)
    axes[0].set_facecolor('#111'); axes[0].tick_params(colors='#aaa')

    # Plot slope vs H
    axes[1].scatter(H_values, slopes, s=80, zorder=5, color='#ffd93d')
    axes[1].plot(H_values, slopes, '--', color='#ffd93d', linewidth=1.5, alpha=0.7)
    for H_v, s in zip(H_values, slopes):
        axes[1].annotate(f"H={H_v}", (H_v, s),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1].set_xlabel("True Hurst exponent H")
    axes[1].set_ylabel("Log-log slope α(H)")
    axes[1].set_title("α(H) is monotone → invertible estimator")
    axes[1].set_facecolor('#111'); axes[1].tick_params(colors='#aaa')
    r, _ = pearsonr(H_values, slopes)
    axes[1].set_title(f"α(H) vs H  (r={r:.3f})")

    plt.tight_layout()
    plt.savefig("hurst_estimation.png", dpi=150, bbox_inches='tight')
    print("[✓] Saved: hurst_estimation.png")
    print(f"    Pearson r(H, slope) = {r:.3f}  (should be close to ±1)")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fbm_approx(n, H=0.7):
    """
    Fast approximate fBm via fractional Gaussian noise (circulant embedding).
    Falls back to correlated Gaussian if size is too small.
    """
    try:
        # Davies-Harte approximation via FFT
        n2 = 2 * n
        k  = np.arange(n2)
        gamma = 0.5 * (np.abs(k - 1)**(2*H) - 2*np.abs(k)**(2*H)
                       + np.abs(k + 1)**(2*H))
        gamma[0] = 1.0
        c = np.fft.fft(gamma)
        c = np.maximum(c.real, 0)  # ensure non-negative
        xi = np.random.randn(n2) + 1j * np.random.randn(n2)
        fgn = np.fft.ifft(np.sqrt(c) * xi).real[:n]
        return np.cumsum(fgn)
    except Exception:
        # Fallback: simple correlated random walk
        z = np.random.randn(n)
        from scipy.ndimage import gaussian_filter1d
        smooth = gaussian_filter1d(z, sigma=max(1, int(n * (H - 0.5) * 4)))
        return np.cumsum(smooth / smooth.std())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # Set dark style globally
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor':   '#111111',
        'axes.edgecolor':   '#555',
        'text.color':       '#cccccc',
        'axes.labelcolor':  '#cccccc',
        'xtick.color':      '#aaaaaa',
        'ytick.color':      '#aaaaaa',
        'grid.color':       '#333',
        'legend.facecolor': '#222',
        'legend.edgecolor': '#555',
    })

    sections = {
        '1': ('Delay embedding visualization',  demo_embedding),
        '2': ('PH of financial data',           demo_ph_financial),
        '3': ('TDA crash detection',            demo_crash_detection),
        '4': ('TDA features for ML',            demo_ml_pipeline),
        '5': ('Topological Hurst estimator',    demo_hurst_estimation),
    }

    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key in sections:
            name, fn = sections[key]
            print(f"\n{'='*55}\n  {name}\n{'='*55}")
            fn()
        else:
            print(f"Unknown section '{key}'. Choose from: {list(sections)}")
    else:
        print("\nTDA for Financial Markets — Demo\n")
        print("Usage:  python demo.py <section>")
        print("        python demo.py 1   # delay embedding")
        print("        python demo.py 2   # PH of financial data")
        print("        python demo.py 3   # crash detection")
        print("        python demo.py 4   # ML pipeline")
        print("        python demo.py 5   # Hurst estimation\n")
        print("Running all sections (may take a few minutes)…\n")
        for k in sorted(sections):
            name, fn = sections[k]
            print(f"\n{'='*55}\n  [{k}] {name}\n{'='*55}")
            try:
                fn()
            except Exception as e:
                print(f"  [!] Error in section {k}: {e}")
