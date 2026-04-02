# Análisis Topológico de Datos para Mercados Financieros

Course materials for **MCD524 — Deep Learning Applied to Finance**  
Instituto Tecnológico de Santo Domingo · Cesar Garcia · 2026

This repository accompanies the lecture on Topological Data Analysis (TDA) applied to financial time series. The core topic is a **topological Hurst estimator** that uses persistent homology to detect market regimes: mean-reverting (H < 0.5), random walk (H ≈ 0.5), or trending (H > 0.5).

## Contents

| Path | Description |
|------|-------------|
| `presentations/slides.html` | Main lecture deck (~30 slides, open in a browser) |
| `presentations/teachers_guide.md` | Slide-by-slide instructor notes |
| `presentations/demo.py` | Interactive 5-section demo |
| `src/tda_financial.py` | Full topological Hurst pipeline |
| `src/rolling_hurst.py` | Classical Hurst (R/S and DFA) for comparison |
| `src/persistence_landscape_demo.py` | Persistence landscape tutorial |
| `src/slide_ph_financial.py` | Minimal code from Slide 25 |
| `src/slide_tda_ml.py` | Minimal code from Slide 26 |

## Getting Started

```bash
# Minimal install
pip install ripser persim yfinance matplotlib numpy scikit-learn pandas

# Run the slide examples
cd src
python slide_ph_financial.py   # Slide 25: persistent homology on S&P 500
python slide_tda_ml.py         # Slide 26: TDA features + ML classifier

# Full topological Hurst analysis
python tda_financial.py --quick        # ~5 min (skips fBm calibration)
python tda_financial.py                # Full run with calibration (~30 min)
python tda_financial.py --ticker ^VIX  # Analyze VIX instead of S&P 500

# Classical Hurst for comparison
python rolling_hurst.py --method rs    # Rescaled range
python rolling_hurst.py --method dfa   # Detrended Fluctuation Analysis
```

Output figures are saved to `src/outputs/`.

## License

Source code (`src/`) is released under the **MIT License**.  
Presentation materials (`presentations/`) are released under **CC BY-NC 4.0** — free to use and adapt with attribution, for non-commercial purposes only.

See [LICENSE](LICENSE) for full terms.
