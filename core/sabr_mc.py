"""Monte Carlo simulation of the SABR stochastic differential equations.

    dF = alpha F^beta dW1,     dalpha = nu alpha dW2,     dW1 dW2 = rho dt

Correlated Gaussian increments:  W1 = Z1,  W2 = rho Z1 + sqrt(1 - rho^2) Z2.

Schemes
-------
"euler"  Standard Euler-Maruyama on both F and alpha:
             F_{i+1}     = F_i + alpha_i (F_i^+)^beta sqrt(dt) W1
             alpha_{i+1} = alpha_i + nu alpha_i sqrt(dt) W2
         Simple but F and alpha can become negative (discretisation bias).

"log"    Production scheme (log transform + full truncation):
             alpha_{i+1} = alpha_i exp(nu sqrt(dt) W2 - nu^2 dt / 2)              (exact, alpha > 0)
             beta = 1 : F_{i+1} = F_i exp(alpha_i sqrt(dt) W1 - alpha_i^2 dt / 2)   (log-Euler, F > 0)
             beta < 1 : F_{i+1} = F_i + alpha_i (F_i^+)^beta sqrt(dt) W1 ; F_T -> max(F_T, 0)
                        (full truncation: only the positive part enters the diffusion)

At maturity F_T = S_T (forward of a maturing contract = spot), so the
simulated terminal forward is the terminal stock price. Under the forward
measure the price is  V_0 = e^{-rT} E[payoff(F_T)].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bs

SCHEMES = {
    "euler": "Standard Euler-Maruyama",
    "log": "Log transform / Full truncation (production standard)",
}


def simulate(F0, T, alpha, beta, rho, nu, n_paths=50_000, n_steps=252, scheme="log",
             antithetic=True, seed=None, n_store=25) -> dict:
    n_paths = int(n_paths)
    n_steps = int(max(1, n_steps))
    if antithetic and n_paths % 2:
        n_paths += 1
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sdt = np.sqrt(dt)
    rho_c = np.sqrt(max(1.0 - rho**2, 0.0))
    half = n_paths // 2 if antithetic else n_paths

    F = np.full(n_paths, float(F0))
    a = np.full(n_paths, float(alpha))
    n_store = min(n_store, n_paths)
    store_F = np.empty((n_store, n_steps + 1))
    store_a = np.empty((n_store, n_steps + 1))
    store_F[:, 0], store_a[:, 0] = F0, alpha
    neg_F_hits = np.zeros(n_paths, dtype=bool)
    neg_a_hits = np.zeros(n_paths, dtype=bool)

    for i in range(n_steps):
        z1 = rng.standard_normal(half)
        z2 = rng.standard_normal(half)
        if antithetic:
            z1 = np.concatenate([z1, -z1])
            z2 = np.concatenate([z2, -z2])
        w1 = z1
        w2 = rho * z1 + rho_c * z2
        if scheme == "euler":
            Fp = np.maximum(F, 0.0)
            F_new = F + a * Fp**beta * sdt * w1
            a_new = a + nu * a * sdt * w2
        else:
            a_new = a * np.exp(nu * sdt * w2 - 0.5 * nu**2 * dt)
            if abs(beta - 1.0) < 1e-12:
                F_new = F * np.exp(a * sdt * w1 - 0.5 * a**2 * dt)
            else:
                Fp = np.maximum(F, 0.0)
                F_new = F + a * Fp**beta * sdt * w1
        F, a = F_new, a_new
        neg_F_hits |= F < 0
        neg_a_hits |= a < 0
        store_F[:, i + 1] = F[:n_store]
        store_a[:, i + 1] = a[:n_store]

    FT = F.copy()
    if scheme != "euler":
        FT = np.maximum(FT, 0.0)
    return {
        "FT": FT, "alphaT": a, "paths_F": store_F, "paths_alpha": store_a,
        "time": np.linspace(0.0, T, n_steps + 1), "n_paths": n_paths, "n_steps": n_steps,
        "dt": dt, "scheme": scheme, "antithetic": antithetic, "F0": float(F0), "T": float(T),
        "pct_negative_F": float(neg_F_hits.mean() * 100.0), "pct_negative_alpha": float(neg_a_hits.mean() * 100.0),
    }


def _pair_means(x, antithetic):
    if antithetic:
        h = x.size // 2
        return 0.5 * (x[:h] + x[h:])
    return x


def price_option(sim: dict, K, r, option="call") -> dict:
    FT, T = sim["FT"], sim["T"]
    disc = np.exp(-r * T)
    pay = np.maximum(FT - K, 0.0) if str(option).lower().startswith("c") else np.maximum(K - FT, 0.0)
    pv = disc * pay
    x = _pair_means(pv, sim["antithetic"])
    price = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(x.size))
    running = np.cumsum(x) / np.arange(1, x.size + 1)
    return {
        "price": price, "std_error": se, "ci_low": price - 1.96 * se, "ci_high": price + 1.96 * se,
        "mean_payoff": float(pay.mean()), "running_mean": running, "itm_prob": float((pay > 0).mean()),
    }


def underlying_stats(sim: dict, r) -> dict:
    FT = sim["FT"]
    x = _pair_means(FT, sim["antithetic"])
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(x.size))
    return {
        "mean_ST": m, "se_ST": se, "F0": sim["F0"], "martingale_error": m - sim["F0"],
        "pv_mean_ST": m * np.exp(-r * sim["T"]),
        "median_ST": float(np.median(FT)), "std_ST": float(FT.std(ddof=1)),
        "q05": float(np.quantile(FT, 0.05)), "q95": float(np.quantile(FT, 0.95)),
    }


def mc_smile(sim: dict, strikes, r, S0) -> pd.DataFrame:
    """Price a strip of strikes with the same paths and back out Black-Scholes implied vols."""
    rows = []
    T = sim["T"]
    for K in strikes:
        opt = "call" if K >= sim["F0"] else "put"
        p = price_option(sim, K, r, opt)
        iv = bs.implied_vol(p["price"], S0, K, T, r, opt)
        rows.append({"Strike": K, "Type": opt.capitalize(), "MC price": p["price"],
                     "Std error": p["std_error"], "MC implied vol": iv})
    return pd.DataFrame(rows)


def convergence_table(sim: dict, K, r, option="call", ref=None) -> pd.DataFrame:
    run = price_option(sim, K, r, option)["running_mean"]
    mult = 2 if sim["antithetic"] else 1
    marks = sorted({int(x) for x in np.unique(np.logspace(2, np.log10(run.size), 8).astype(int)) if x <= run.size})
    rows = []
    for k in marks:
        row = {"Paths": k * mult, "MC price": run[k - 1]}
        if ref is not None:
            row["Error vs Hagan"] = run[k - 1] - ref
        rows.append(row)
    return pd.DataFrame(rows)
