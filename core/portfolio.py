"""Valuation of a portfolio of European stock options (Module 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bs

INPUT_COLUMNS = [
    "Ticker", "Spot", "Type", "Position", "Quantity", "Strike",
    "Maturity (years)", "Volatility (%)", "Market price", "Premium paid/received",
]


def default_positions() -> pd.DataFrame:
    return pd.DataFrame([
        {"Ticker": "AAPL", "Spot": 230.0, "Type": "Call", "Position": "Long", "Quantity": 100,
         "Strike": 230.0, "Maturity (years)": 0.25, "Volatility (%)": 25.0,
         "Market price": 0.0, "Premium paid/received": 11.5},
        {"Ticker": "AAPL", "Spot": 230.0, "Type": "Put", "Position": "Short", "Quantity": 50,
         "Strike": 210.0, "Maturity (years)": 0.25, "Volatility (%)": 27.0,
         "Market price": 0.0, "Premium paid/received": 4.0},
        {"Ticker": "MSFT", "Spot": 510.0, "Type": "Call", "Position": "Short", "Quantity": 100,
         "Strike": 540.0, "Maturity (years)": 0.5, "Volatility (%)": 24.0,
         "Market price": 0.0, "Premium paid/received": 25.0},
    ])


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in INPUT_COLUMNS:
        if c not in d:
            d[c] = 0.0 if c not in ("Ticker", "Type", "Position") else ""
    d = d.dropna(how="all")
    num = ["Spot", "Quantity", "Strike", "Maturity (years)", "Volatility (%)", "Market price", "Premium paid/received"]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["Type"] = d["Type"].fillna("Call").astype(str)
    d["Position"] = d["Position"].fillna("Long").astype(str)
    d["Ticker"] = d["Ticker"].fillna("").astype(str).str.upper().str.strip()
    return d.reset_index(drop=True)


def validate(df: pd.DataFrame) -> list[str]:
    errs = []
    d = _clean(df)
    for i, row in d.iterrows():
        tag = f"Row {i + 1} ({row['Ticker'] or 'no ticker'})"
        if row["Spot"] <= 0:
            errs.append(f"{tag}: spot must be > 0.")
        if row["Strike"] <= 0:
            errs.append(f"{tag}: strike must be > 0.")
        if row["Quantity"] < 0:
            errs.append(f"{tag}: quantity must be >= 0 (use Position = Short for short).")
        if row["Maturity (years)"] < 0:
            errs.append(f"{tag}: maturity must be >= 0.")
        if row["Volatility (%)"] <= 0 and row["Maturity (years)"] > 0:
            errs.append(f"{tag}: volatility must be > 0.")
    return errs


def value_portfolio(df: pd.DataFrame, r: float) -> tuple[pd.DataFrame, dict]:
    """Return a per-position result table and portfolio totals.

    Per option (one option = right on one share):
        intrinsic = max(S-K,0) or max(K-S,0)
        model value = Black-Scholes price
        mark = market price if provided (>0), else model value
    Position values = sign * quantity * per-option value  (sign = +1 long, -1 short)
    """
    d = _clean(df)
    rows = []
    for _, p in d.iterrows():
        sign = 1.0 if p["Position"].lower().startswith("l") else -1.0
        S, K, T = p["Spot"], p["Strike"], p["Maturity (years)"]
        sig = p["Volatility (%)"] / 100.0
        q = p["Quantity"]
        opt = p["Type"]
        g = bs.greeks(S, K, T, r, sig, opt)
        intr = bs.intrinsic(S, K, opt)
        mkt = p["Market price"]
        mark = mkt if mkt > 0 else g["price"]
        source = "Market quote" if mkt > 0 else "Black-Scholes"
        mv = sign * q * mark
        prem = p["Premium paid/received"]
        cost = sign * q * prem
        moneyness = (S - K) if opt.lower().startswith("c") else (K - S)
        status = "ITM" if moneyness > 1e-9 else ("ATM" if abs(moneyness) <= 1e-9 else "OTM")
        rows.append({
            "Ticker": p["Ticker"], "Type": opt, "Position": p["Position"], "Quantity": q,
            "Spot": S, "Strike": K, "T (y)": T, "Vol (%)": p["Volatility (%)"],
            "Moneyness": status,
            "Intrinsic / option": intr,
            "BS value / option": g["price"],
            "Time value / option": g["price"] - intr,
            "Mark / option": mark,
            "Valuation source": source,
            "Intrinsic value (position)": sign * q * intr,
            "Market value (position)": mv,
            "Cost basis (position)": cost,
            "Unrealized P&L": mv - cost if prem > 0 else np.nan,
            "Delta (shares)": sign * q * g["delta"],
            "Gamma": sign * q * g["gamma"],
            "Vega (per 1 vol pt)": sign * q * g["vega"] / 100.0,
            "Theta (per day)": sign * q * g["theta"] / 365.0,
            "Rho (per 1% rate)": sign * q * g["rho"] / 100.0,
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res, {}
    totals = {
        "Market value": float(res["Market value (position)"].sum()),
        "Intrinsic value": float(res["Intrinsic value (position)"].sum()),
        "Time value": float(res["Market value (position)"].sum() - res["Intrinsic value (position)"].sum()),
        "Cost basis": float(res["Cost basis (position)"].sum()),
        "Unrealized P&L": float(np.nansum(res["Unrealized P&L"].values)),
        "Delta (shares)": float(res["Delta (shares)"].sum()),
        "Gamma": float(res["Gamma"].sum()),
        "Vega (per 1 vol pt)": float(res["Vega (per 1 vol pt)"].sum()),
        "Theta (per day)": float(res["Theta (per day)"].sum()),
        "Rho (per 1% rate)": float(res["Rho (per 1% rate)"].sum()),
        "Long market value": float(res.loc[res["Market value (position)"] > 0, "Market value (position)"].sum()),
        "Short market value": float(res.loc[res["Market value (position)"] < 0, "Market value (position)"].sum()),
    }
    return res, totals


def by_underlying(res: pd.DataFrame) -> pd.DataFrame:
    if res.empty:
        return res
    cols = ["Market value (position)", "Intrinsic value (position)", "Delta (shares)", "Gamma",
            "Vega (per 1 vol pt)", "Theta (per day)"]
    g = res.groupby("Ticker")[cols].sum().reset_index()
    spots = res.groupby("Ticker")["Spot"].first().values
    g.insert(1, "Spot", spots)
    g["Delta exposure ($)"] = g["Delta (shares)"] * g["Spot"]
    return g


def value_profile(df: pd.DataFrame, r: float, ticker: str, n: int = 121, width: float = 0.4) -> pd.DataFrame:
    """Portfolio value vs. the spot of one underlying (other underlyings held fixed).

    Returns columns: Spot, Value today (BS), Intrinsic value.
    """
    d = _clean(df)
    d = d[d["Ticker"] == ticker]
    if d.empty:
        return pd.DataFrame()
    S0 = float(d["Spot"].iloc[0])
    grid = np.linspace(S0 * (1 - width), S0 * (1 + width), n)
    today = np.zeros_like(grid)
    intr = np.zeros_like(grid)
    for _, p in d.iterrows():
        sign = 1.0 if p["Position"].lower().startswith("l") else -1.0
        q = p["Quantity"]
        today += sign * q * bs.price(grid, p["Strike"], p["Maturity (years)"], r, p["Volatility (%)"] / 100.0, p["Type"])
        intr += sign * q * bs.intrinsic(grid, p["Strike"], p["Type"])
    return pd.DataFrame({"Spot": grid, "Value today (BS)": today, "Intrinsic value": intr})
