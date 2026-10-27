"""Black-Scholes-Merton pricing engine for European options on non-dividend-paying stocks.

All functions are vectorised (NumPy broadcasting) and return Python floats when
called with scalar inputs.

Conventions
-----------
S      spot price of the underlying
K      strike
T      time to maturity in years
r      continuously compounded risk-free rate (decimal, e.g. 0.05)
sigma  annualised volatility (decimal, e.g. 0.20)
option "call" or "put"
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_EPS = 1e-12


def _out(x):
    """Return a Python float for 0-d results, otherwise the array."""
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def _is_call(option) -> bool:
    return str(option).strip().lower().startswith("c")


def d1_d2(S, K, T, r, sigma):
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))
    Tp = np.maximum(T, _EPS)
    sp = np.maximum(sigma, _EPS)
    sqrtT = np.sqrt(Tp)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sp**2) * Tp) / (sp * sqrtT)
    d2 = d1 - sp * sqrtT
    return d1, d2


def price(S, K, T, r, sigma, option="call"):
    """Black-Scholes price. At or after maturity the intrinsic value is returned."""
    S, K, T = (np.asarray(v, dtype=float) for v in (S, K, T))
    d1, d2 = d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * np.maximum(T, 0.0))
    call = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    if _is_call(option):
        val = call
        intrinsic = np.maximum(S - K, 0.0)
    else:
        val = call - S + K * disc  # put-call parity
        intrinsic = np.maximum(K - S, 0.0)
    val = np.where(T <= _EPS, intrinsic, val)
    return _out(np.maximum(val, 0.0))


def intrinsic(S, K, option="call"):
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)
    return _out(np.maximum(S - K, 0.0) if _is_call(option) else np.maximum(K - S, 0.0))


def delta(S, K, T, r, sigma, option="call"):
    S, K, T = (np.asarray(v, dtype=float) for v in (S, K, T))
    d1, _ = d1_d2(S, K, T, r, sigma)
    if _is_call(option):
        d = norm.cdf(d1)
        expired = np.where(S > K, 1.0, 0.0)
    else:
        d = norm.cdf(d1) - 1.0
        expired = np.where(S < K, -1.0, 0.0)
    return _out(np.where(T <= _EPS, expired, d))


def gamma(S, K, T, r, sigma, option="call"):
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))
    d1, _ = d1_d2(S, K, T, r, sigma)
    g = norm.pdf(d1) / (S * np.maximum(sigma, _EPS) * np.sqrt(np.maximum(T, _EPS)))
    return _out(np.where(T <= _EPS, 0.0, g))


def vega(S, K, T, r, sigma, option="call"):
    """Sensitivity to a change of 1.00 (i.e. 100 vol points) in sigma."""
    S, K, T = (np.asarray(v, dtype=float) for v in (S, K, T))
    d1, _ = d1_d2(S, K, T, r, sigma)
    v = S * norm.pdf(d1) * np.sqrt(np.maximum(T, 0.0))
    return _out(np.where(T <= _EPS, 0.0, v))


def theta(S, K, T, r, sigma, option="call"):
    """Time decay per year (divide by 365 for a calendar day)."""
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))
    d1, d2 = d1_d2(S, K, T, r, sigma)
    Tp = np.maximum(T, _EPS)
    first = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(Tp))
    disc = np.exp(-r * Tp)
    if _is_call(option):
        th = first - r * K * disc * norm.cdf(d2)
    else:
        th = first + r * K * disc * norm.cdf(-d2)
    return _out(np.where(T <= _EPS, 0.0, th))


def rho(S, K, T, r, sigma, option="call"):
    S, K, T = (np.asarray(v, dtype=float) for v in (S, K, T))
    _, d2 = d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * np.maximum(T, 0.0))
    if _is_call(option):
        rh = K * T * disc * norm.cdf(d2)
    else:
        rh = -K * T * disc * norm.cdf(-d2)
    return _out(np.where(T <= _EPS, 0.0, rh))


def greeks(S, K, T, r, sigma, option="call") -> dict:
    return {
        "price": price(S, K, T, r, sigma, option),
        "delta": delta(S, K, T, r, sigma, option),
        "gamma": gamma(S, K, T, r, sigma, option),
        "vega": vega(S, K, T, r, sigma, option),
        "theta": theta(S, K, T, r, sigma, option),
        "rho": rho(S, K, T, r, sigma, option),
    }


def replicating_portfolio(S, K, T, r, sigma, option="call") -> dict:
    """Static snapshot of the Black-Scholes replicating portfolio.

    option = shares * S + bond, with bond > 0 meaning lending (long bond)
    and bond < 0 meaning borrowing (short bond).
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)
    if _is_call(option):
        shares = float(norm.cdf(d1))
        bond = float(-K * disc * norm.cdf(d2))
    else:
        shares = float(norm.cdf(d1) - 1.0)
        bond = float(K * disc * norm.cdf(-d2))
    return {"shares": shares, "bond": bond, "value": shares * S + bond}


def implied_vol(market_price, S, K, T, r, option="call", lo=1e-4, hi=5.0):
    """Invert Black-Scholes with Brent's method. Returns NaN if no solution."""
    try:
        market_price = float(market_price)
        if T <= 0 or market_price <= 0:
            return float("nan")
        disc = np.exp(-r * T)
        lower = max(S - K * disc, 0.0) if _is_call(option) else max(K * disc - S, 0.0)
        upper = S if _is_call(option) else K * disc
        if market_price <= lower + 1e-10 or market_price >= upper - 1e-10:
            return float("nan")
        f = lambda s: price(S, K, T, r, s, option) - market_price
        if f(lo) * f(hi) > 0:
            return float("nan")
        return float(brentq(f, lo, hi, xtol=1e-10, maxiter=200))
    except Exception:
        return float("nan")


# ---------- Black (1976) on forwards: used by SABR -----------------------
def black76(F, K, T, r, sigma, option="call"):
    """Discounted Black-76 price. Equals Black-Scholes when F = S*exp(rT)."""
    F, K, T, sigma = (np.asarray(v, dtype=float) for v in (F, K, T, sigma))
    Tp = np.maximum(T, _EPS)
    sp = np.maximum(sigma, _EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sp**2 * Tp) / (sp * np.sqrt(Tp))
    d2 = d1 - sp * np.sqrt(Tp)
    disc = np.exp(-r * np.maximum(T, 0.0))
    if _is_call(option):
        v = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        intr = disc * np.maximum(F - K, 0.0)
    else:
        v = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        intr = disc * np.maximum(K - F, 0.0)
    return _out(np.where(T <= _EPS, intr, np.maximum(v, 0.0)))


def black76_delta(F, K, T, r, sigma, option="call"):
    """Delta with respect to the SPOT when F = S*exp(rT) (= BS delta)."""
    F, K, T, sigma = (np.asarray(v, dtype=float) for v in (F, K, T, sigma))
    Tp = np.maximum(T, _EPS)
    sp = np.maximum(sigma, _EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sp**2 * Tp) / (sp * np.sqrt(Tp))
    d = norm.cdf(d1) if _is_call(option) else norm.cdf(d1) - 1.0
    return _out(d)
