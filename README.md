# Option Analytics Lab

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://option-analytics-lab.streamlit.app)

**Live app (computer, Android, iPhone): https://option-analytics-lab.streamlit.app**

A Python application (Streamlit) for European options on non-dividend-paying stocks, with four modules:

| # | Module | What it does |
|---|--------|--------------|
| 1 | **Portfolio Evaluation** | Market value, intrinsic value, time value and Greeks of a portfolio of calls/puts (long/short). Spot prices fetched online (Yahoo Finance) or typed manually. |
| 2 | **Arbitrage Calculator** | Synthetic options from stock + bonds (put-call parity, no-arbitrage bounds, Black-Scholes replication). Prints the step-by-step strategy and verifies it with the payoff at maturity. |
| 3 | **Delta / Delta-Gamma Hedging** | Daily hedge adjustments on historical prices or on a Spot Price Simulator (1-5 days), P&L projection, Monte Carlo P&L distribution, charts, printable daily reports. |
| 4 | **SABR Calculator** | Hagan implied volatility, calibration of α, ρ, ν (β = 1) from CSV/Excel (option quotes or historical prices), smile across spot/forward levels, SABR vs Black-Scholes prices, mispricing strategies, butterfly check, and **Monte Carlo simulation of the SABR SDEs** (Euler-Maruyama or log transform / full truncation, user-chosen number of simulations). |

Every module has a **❓ Help** button (theory, assumptions, formulas generated with **SymPy**), a printable **PDF report** and an **Excel export**.

---

## 1. Run it on a computer (Windows / macOS / Linux)

Requirements: **Python 3.9 or newer** (Anaconda works).

**Windows:** double-click `run_app.bat`.
**macOS / Linux:** `./run_app.sh`

Or manually, from this folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in the browser at `http://localhost:8501`. Stop it with `Ctrl+C` in the terminal.

> Live prices need an internet connection. Without it, every input can be entered manually (the app tells you when a download fails and keeps the manual value).

## 2. Use it on a phone (Android / iPhone)

The same code is deployed as a web app on Streamlit Community Cloud and opened from the phone browser, where it can be added to the home screen like an app. See **[DEPLOY_MOBILE.md](DEPLOY_MOBILE.md)**.

## 3. Sample data (folder `data/`)

| File | Use |
|------|-----|
| `sample_sabr_option_quotes.csv` / `.xlsx` | SABR calibration from option quotes (Strike, Type, Market Price, Implied Vol, Maturity, Spot, Rate) |
| `sample_sabr_prices_only.csv` | Same quotes without implied vols: the app inverts Black-Scholes itself |
| `sample_historical_prices.csv` | 5 years of daily closes (Date, Close) for the historical SABR calibration and the historical hedging backtest |

## 4. Project structure

```
app.py                  entry point (navigation)
views/                  one page per module (user interface)
core/                   calculation engine, independent of the interface
  bs.py                 Black-Scholes, Greeks, implied vol, Black-76
  portfolio.py          module 1
  arbitrage.py          module 2 (legs, strategies, payoff verification)
  hedging.py            module 3 (delta / delta-gamma, GBM simulator, Monte Carlo)
  sabr.py               module 4 (Hagan, calibration, smile grid, density check)
  sabr_mc.py            Monte Carlo of the SABR SDEs
  help_content.py       Help pages, formulas built with SymPy
  reports.py            PDF (fpdf2 + matplotlib) and Excel exports
  market.py             Yahoo Finance data (yfinance) with graceful fallback
ui/components.py        shared widgets (Help dialog, live/manual spot input, downloads)
tests/test_core.py      numerical checks of the engine
data/                   sample files
```

## 5. Tests

```bash
python tests/test_core.py     # numerical checks of the engine
python tests/smoke_ui.py      # runs every page headless (buttons, uploads, PDF/Excel reports), offline
```

15 checks: Black-Scholes reference values and Greeks (finite differences), implied-vol round trip, every arbitrage strategy has positive cash today and a zero / non-negative payoff in all scenarios, hedging reduces P&L dispersion (delta-gamma < delta < unhedged), Hagan limits (ν → 0, ATM continuity), calibration recovers known parameters, Monte Carlo matches Black-Scholes when ν = 0 and Hagan's formula otherwise, historical estimator recovers ν and ρ on simulated SABR paths.

## 6. Main assumptions

European options, no dividends, constant risk-free rate (continuous compounding), F = S·e^(rT), maturities in years, one option on one share, daily re-balancing on closing prices (252 trading days per year by default). Details and formulas in each module's Help.
