"""Online market data (Yahoo Finance via yfinance) with graceful failure.

Every function raises MarketDataError with a readable message when data cannot
be retrieved, so the UI can fall back to manual input.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd


class MarketDataError(RuntimeError):
    pass


def _yf():
    try:
        import yfinance as yf  # imported lazily so the app runs without it
    except ImportError as exc:  # pragma: no cover
        raise MarketDataError("yfinance is not installed (pip install yfinance).") from exc
    return yf


def fetch_spot(ticker: str) -> tuple[float, str]:
    """Return (last price, description)."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise MarketDataError("Enter a ticker symbol first (e.g. AAPL).")
    yf = _yf()
    try:
        t = yf.Ticker(ticker)
        px = None
        try:
            px = t.fast_info.get("last_price") if hasattr(t.fast_info, "get") else t.fast_info["last_price"]
        except Exception:
            px = None
        if px is None or not np.isfinite(px) or px <= 0:
            hist = t.history(period="5d", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                raise MarketDataError(f"No price data returned for '{ticker}'.")
            px = float(hist["Close"].dropna().iloc[-1])
        px = float(px)
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(f"Could not fetch '{ticker}': {exc}") from exc
    if not math.isfinite(px) or px <= 0:
        raise MarketDataError(f"Invalid price received for '{ticker}'.")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return px, f"Yahoo Finance, {ticker}, {stamp}"


def fetch_history(ticker: str, period: str = "1y") -> pd.Series:
    """Daily closing prices (index = dates)."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise MarketDataError("Enter a ticker symbol first.")
    yf = _yf()
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    except Exception as exc:
        raise MarketDataError(f"Could not fetch history for '{ticker}': {exc}") from exc
    if hist is None or hist.empty or "Close" not in hist:
        raise MarketDataError(f"No history returned for '{ticker}'.")
    s = hist["Close"].dropna().astype(float)
    try:
        s.index = pd.to_datetime(s.index).tz_localize(None)
    except (TypeError, AttributeError):
        s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s


def historical_volatility(prices: pd.Series, window: int | None = None, periods_per_year: int = 252) -> float:
    """Annualised close-to-close volatility of log returns."""
    lr = np.log(prices / prices.shift(1)).dropna()
    if window:
        lr = lr.iloc[-window:]
    if len(lr) < 2:
        raise MarketDataError("Not enough data to estimate volatility.")
    return float(lr.std(ddof=1) * np.sqrt(periods_per_year))


def fetch_option_expiries(ticker: str) -> list[str]:
    yf = _yf()
    try:
        exps = list(yf.Ticker(ticker.strip().upper()).options)
    except Exception as exc:
        raise MarketDataError(f"Could not fetch option expiries for '{ticker}': {exc}") from exc
    if not exps:
        raise MarketDataError(f"No listed options found for '{ticker}'.")
    return exps


def fetch_option_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (calls, puts) with columns strike, bid, ask, lastPrice, mid, impliedVolatility."""
    yf = _yf()
    try:
        ch = yf.Ticker(ticker.strip().upper()).option_chain(expiry)
    except Exception as exc:
        raise MarketDataError(f"Could not fetch option chain: {exc}") from exc
    out = []
    for df in (ch.calls, ch.puts):
        d = df[["strike", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest"]].copy()
        mid = (d["bid"] + d["ask"]) / 2.0
        d["mid"] = np.where((d["bid"] > 0) & (d["ask"] > 0), mid, d["lastPrice"])
        out.append(d.reset_index(drop=True))
    return out[0], out[1]
