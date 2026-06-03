# Continuous‑Time Markov Jump Engine for ETFs

Models transitions to high‑volatility states as a continuous‑time Markov jump process. The jump intensity depends on current macro variables (VIX, DXY, yields) via a logistic regression. The per‑ETF score is the predicted probability of entering a high‑volatility state.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- All available macro variables as covariates
- Logistic regression on binary indicator |return| > threshold
- Score = conditional probability of high‑volatility day given macro
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-continuous-time-markov-jump-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High jump intensity → the ETF is likely to experience a large move given current macro conditions (VIX high, etc.).
- Low intensity → ETF is stable, low probability of extreme returns.

## Requirements

See `requirements.txt`.
