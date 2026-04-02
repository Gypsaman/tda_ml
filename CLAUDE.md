# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational and research project on **Topological Data Analysis (TDA) applied to financial time series**, taught as MCD524: Deep Learning Applied to Finance at Instituto Tecnológico de Santo Domingo. The core innovation is a **topological Hurst estimator** that uses persistent homology to detect market regimes (mean-reverting H < 0.5, random walk H ≈ 0.5, trending H > 0.5).

## Running Scripts

```bash
# Minimal install (for slide examples)
pip install ripser persim yfinance matplotlib numpy scikit-learn pandas

# Full install (all demos)
pip install ripser persim giotto-tda gudhi yfinance numpy scipy matplotlib scikit-learn pandas

# Simple slide examples
cd src && python slide_ph_financial.py   # Slide 25: basic PH on S&P 500
cd src && python slide_tda_ml.py         # Slide 26: TDA features + ML classifier

# Full topological Hurst analysis
python src/tda_financial.py                     # Full run (30+ min, includes fBm calibration)
python src/tda_financial.py --quick             # Skip calibration (~5 min)
python src/tda_financial.py --ticker ^VIX       # Analyze VIX
python src/tda_financial.py --no-save           # Interactive (don't save figures)

# Classical Hurst comparison
python src/rolling_hurst.py --method dfa        # Detrended Fluctuation Analysis
python src/rolling_hurst.py --method rs         # R/S analysis (default)

# Educational demo
python presentations/demo.py
```

Output figures are saved to `src/outputs/`.

## Architecture

The pipeline has four stages:

1. **Mathematical foundations** (`tda_financial.py`): fBm simulation via Davies-Harte/circulant embedding → sliding-window delay embedding (1D series → point cloud in ℝᵈ) → Vietoris-Rips complex construction.

2. **Topological computation** (`tda_financial.py`): Persistent homology (H₀, H₁) via Ripser (primary) or GUDHI (fallback, graceful try/except at module level) → barcode count function → power-law slope α(H) fitting.

3. **Calibration & inversion** (`tda_financial.py`): Monte Carlo over H ∈ [0.1, 0.9] with 10 fBm trials per value establishes α(H) lookup table; `make_alpha_inverter()` interpolates H from estimated α.

4. **Financial application** (`tda_financial.py`, `rolling_hurst.py`): Rolling window analysis with 5-day step; dual estimation of H_top (topological) vs H_RS (classical R/S); total H₁ persistence as turbulence indicator.

### Key modules

- **`src/tda_financial.py`** (~870 lines) — flagship research module; complete topological Hurst pipeline with CLI (`--ticker`, `--period`, `--window`, `--quick`, `--no-save`)
- **`src/rolling_hurst.py`** (~346 lines) — standalone classical Hurst (R/S and DFA methods) with rolling window and CLI
- **`src/persistence_landscape_demo.py`** (~336 lines) — educational tool explaining persistence landscape vectorization
- **`presentations/demo.py`** — 5-section interactive demo (delay embedding, financial PH, crash detection, ML pipeline, Hurst comparison)
- **`presentations/slides.html`** — main ~30-slide HTML presentation deck
- **`presentations/teachers_guide.md`** — slide-by-slide instructor notes with talking points, student Q&A, and timing

### Design conventions

- **Backend abstraction**: Ripser is preferred; GUDHI is the fallback. Both are wrapped at module import level.
- **Numpy-first**: All numerical operations avoid unnecessary dependencies.
- **Pure functions**: Each pipeline step (PH → barcode → α → H) is a standalone function.
- **Matplotlib with GridSpec**: Multi-panel publication-quality figures throughout.

## Dependencies

Core: `numpy`, `pandas`, `matplotlib`, `scipy`, `yfinance`, `ripser`, `persim`, `scikit-learn`  
Optional: `gudhi` (PH fallback), `giotto-tda` (advanced features)
