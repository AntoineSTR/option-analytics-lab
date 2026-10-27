"""Delta and Delta-Gamma hedging engine (Module 3).

The book holds q = sign * N options (option 1). Each day the hedger re-balances:

Delta hedge         : h_t = -q * Delta1_t shares
Delta-Gamma hedge   : w_t = -q * Gamma1_t / Gamma2_t units of option 2,
                      h_t = -(q * Delta1_t + w_t * Delta2_t) shares

All trades are financed through a cash account earning r (self-financing).
Portfolio value Pi_t = option positions + shares + cash, with Pi_0 = 0, so Pi_t
is directly the cumulative P&L of the hedged book.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import bs


@dataclass
class HedgeSpec:
    K: float
    T: float                   # maturity (years) at day 0
    r: float
    sigma: float               # volatility used for pricing / hedge ratios
    option: str = "call"
    position: str = "Short"    # position held in option 1
    n: float = 100.0           # number of options (each on 1 share)
    K2: float = 0.0            # hedge option (delta-gamma)
    T2: float = 0.0
    option2: str = "call"
    days_per_year: int = 252
    tc_bps: float = 0.0        # transaction cost on traded notional (basis points)

    @property
    def q(self) -> float:
        return (1.0 if str(self.position).lower().startswith("l") else -1.0) * self.n


def simulate_gbm(S0, mu, sigma, n_days, n_paths=1, seed=None, days_per_year=252):
    """Exact GBM: S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z). Shape (n_paths, n_days+1)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / days_per_year
    z = rng.standard_normal((n_paths, n_days))
    incr = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    logp = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return S0 * np.exp(logp)


def _core(S: np.ndarray, spec: HedgeSpec) -> dict:
    """Vectorised hedging over paths. S has shape (m, n+1)."""
    S = np.atleast_2d(np.asarray(S, dtype=float))
    m, n1 = S.shape
    dt = 1.0 / spec.days_per_year
    tgrid = np.arange(n1) * dt
    tau1 = np.maximum(spec.T - tgrid, 0.0)[None, :].repeat(m, 0)
    tau2 = np.maximum(spec.T2 - tgrid, 0.0)[None, :].repeat(m, 0)
    q = spec.q
    tc = spec.tc_bps / 1e4
    growth = np.exp(spec.r * dt)

    V1 = bs.price(S, spec.K, tau1, spec.r, spec.sigma, spec.option)
    D1 = bs.delta(S, spec.K, tau1, spec.r, spec.sigma, spec.option)
    G1 = bs.gamma(S, spec.K, tau1, spec.r, spec.sigma, spec.option)
    have2 = spec.K2 > 0 and spec.T2 > 0
    if have2:
        V2 = bs.price(S, spec.K2, tau2, spec.r, spec.sigma, spec.option2)
        D2 = bs.delta(S, spec.K2, tau2, spec.r, spec.sigma, spec.option2)
        G2 = bs.gamma(S, spec.K2, tau2, spec.r, spec.sigma, spec.option2)
    else:
        V2 = D2 = G2 = np.zeros_like(S)

    # --- unhedged ---------------------------------------------------------
    cash_u = np.zeros_like(S)
    cash_u[:, 0] = -q * V1[:, 0]
    for t in range(1, n1):
        cash_u[:, t] = cash_u[:, t - 1] * growth
    pi_u = q * V1 + cash_u

    # --- delta hedge ------------------------------------------------------
    h = -q * D1
    trade_h = np.diff(np.concatenate([np.zeros((m, 1)), h], axis=1), axis=1)
    cost_d = np.abs(trade_h) * S * tc
    cash_d = np.zeros_like(S)
    cash_d[:, 0] = -(q * V1[:, 0] + h[:, 0] * S[:, 0]) - cost_d[:, 0]
    for t in range(1, n1):
        cash_d[:, t] = cash_d[:, t - 1] * growth - trade_h[:, t] * S[:, t] - cost_d[:, t]
    pi_d = q * V1 + h * S + cash_d

    # --- delta-gamma hedge ------------------------------------------------
    if have2:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(G2) > 1e-12, -q * G1 / G2, 0.0)
        hg = -(q * D1 + w * D2)
        trade_w = np.diff(np.concatenate([np.zeros((m, 1)), w], axis=1), axis=1)
        trade_hg = np.diff(np.concatenate([np.zeros((m, 1)), hg], axis=1), axis=1)
        cost_g = (np.abs(trade_hg) * S + np.abs(trade_w) * V2) * tc
        cash_g = np.zeros_like(S)
        cash_g[:, 0] = -(q * V1[:, 0] + w[:, 0] * V2[:, 0] + hg[:, 0] * S[:, 0]) - cost_g[:, 0]
        for t in range(1, n1):
            cash_g[:, t] = (cash_g[:, t - 1] * growth - trade_hg[:, t] * S[:, t]
                            - trade_w[:, t] * V2[:, t] - cost_g[:, t])
        pi_g = q * V1 + w * V2 + hg * S + cash_g
    else:
        w = hg = trade_w = trade_hg = cash_g = pi_g = None

    return dict(S=S, tau1=tau1, tau2=tau2, V1=V1, D1=D1, G1=G1, V2=V2, D2=D2, G2=G2,
                pi_u=pi_u, h=h, trade_h=trade_h, cash_d=cash_d, pi_d=pi_d,
                w=w, hg=hg, trade_w=trade_w, trade_hg=trade_hg, cash_g=cash_g, pi_g=pi_g, have2=have2)


def _instr(qty, unit):
    if abs(qty) < 1e-9:
        return "No trade"
    return f"{'Buy' if qty > 0 else 'Sell'} {abs(qty):,.2f} {unit}"


def run_hedge(spots, spec: HedgeSpec, dates=None) -> dict:
    """Single path -> daily adjustment reports (DataFrames) + summary."""
    spots = np.asarray(spots, dtype=float).ravel()
    c = _core(spots[None, :], spec)
    n1 = len(spots)
    day = np.arange(n1)
    labels = list(dates) if dates is not None else [f"Day {i}" for i in day]

    def g(a):
        return None if a is None else a[0]

    base = pd.DataFrame({
        "Day": day, "Date": labels, "Spot": spots,
        "Time to maturity (y)": g(c["tau1"]),
        "Option value": g(c["V1"]), "Delta": g(c["D1"]), "Gamma": g(c["G1"]),
    })

    pi_u = g(c["pi_u"])
    unhedged = base.copy()
    unhedged["Position value"] = spec.q * g(c["V1"])
    unhedged["Cumulative P&L"] = pi_u
    unhedged["Daily P&L"] = np.diff(np.concatenate([[0.0], pi_u]))

    pi_d = g(c["pi_d"])
    delta_df = base.copy()
    delta_df["Position delta (q*Delta)"] = spec.q * g(c["D1"])
    delta_df["Shares held"] = g(c["h"])
    delta_df["Shares traded"] = g(c["trade_h"])
    delta_df["Adjustment"] = [_instr(x, "shares") for x in g(c["trade_h"])]
    delta_df["Trade cash flow"] = -g(c["trade_h"]) * spots
    delta_df["Cash account"] = g(c["cash_d"])
    delta_df["Portfolio value"] = pi_d
    delta_df["Daily P&L"] = np.diff(np.concatenate([[0.0], pi_d]))
    delta_df["Cumulative P&L"] = pi_d
    delta_df["Unhedged cum. P&L"] = pi_u

    gamma_df = None
    if c["have2"]:
        pi_g = g(c["pi_g"])
        gamma_df = base.copy()
        gamma_df["Option 2 value"] = g(c["V2"])
        gamma_df["Delta 2"] = g(c["D2"])
        gamma_df["Gamma 2"] = g(c["G2"])
        gamma_df["Option 2 held"] = g(c["w"])
        gamma_df["Option 2 traded"] = g(c["trade_w"])
        gamma_df["Shares held"] = g(c["hg"])
        gamma_df["Shares traded"] = g(c["trade_hg"])
        gamma_df["Adjustment"] = [
            f"{_instr(a, 'hedge options')}; {_instr(b, 'shares')}"
            for a, b in zip(g(c["trade_w"]), g(c["trade_hg"]))
        ]
        gamma_df["Net delta"] = spec.q * g(c["D1"]) + g(c["w"]) * g(c["D2"]) + g(c["hg"])
        gamma_df["Net gamma"] = spec.q * g(c["G1"]) + g(c["w"]) * g(c["G2"])
        gamma_df["Cash account"] = g(c["cash_g"])
        gamma_df["Portfolio value"] = pi_g
        gamma_df["Daily P&L"] = np.diff(np.concatenate([[0.0], pi_g]))
        gamma_df["Cumulative P&L"] = pi_g

    summary = {
        "Days": n1 - 1,
        "Spot start": spots[0], "Spot end": spots[-1],
        "Spot move (%)": (spots[-1] / spots[0] - 1) * 100,
        "Option value start": g(c["V1"])[0], "Option value end": g(c["V1"])[-1],
        "Unhedged P&L": pi_u[-1],
        "Delta-hedged P&L": pi_d[-1],
    }
    if gamma_df is not None:
        summary["Delta-gamma hedged P&L"] = gamma_df["Cumulative P&L"].iloc[-1]
    return {"unhedged": unhedged, "delta": delta_df, "gamma": gamma_df, "summary": summary}


def mc_pnl(spec: HedgeSpec, S0, mu, sigma_real, n_days, n_paths=2000, seed=None) -> pd.DataFrame:
    """Distribution of final P&L over simulated GBM paths for each strategy."""
    paths = simulate_gbm(S0, mu, sigma_real, n_days, n_paths, seed, spec.days_per_year)
    c = _core(paths, spec)
    out = {"Final spot": paths[:, -1], "Unhedged": c["pi_u"][:, -1], "Delta hedge": c["pi_d"][:, -1]}
    if c["have2"]:
        out["Delta-gamma hedge"] = c["pi_g"][:, -1]
    return pd.DataFrame(out)


def pnl_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in df.columns if c != "Final spot"]:
        x = df[col].values
        rows.append({"Strategy": col, "Mean": x.mean(), "Std dev": x.std(ddof=1),
                     "5% quantile (VaR 95)": np.quantile(x, 0.05), "Min": x.min(), "Max": x.max()})
    return pd.DataFrame(rows)
