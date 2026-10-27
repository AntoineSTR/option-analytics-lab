"""Numerical checks of the core engine. Run:  python -m pytest -q   (or python tests/test_core.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import arbitrage, bs, hedging, portfolio, sabr, sabr_mc  # noqa: E402


def close(a, b, tol):
    assert abs(a - b) < tol, f"{a} vs {b} (tol {tol})"


def test_black_scholes_reference():
    close(bs.price(100, 100, 1, 0.05, 0.2, "call"), 10.450584, 1e-5)
    close(bs.price(100, 100, 1, 0.05, 0.2, "put"), 5.573526, 1e-5)
    close(bs.delta(100, 100, 1, 0.05, 0.2, "call"), 0.636831, 1e-5)
    close(bs.gamma(100, 100, 1, 0.05, 0.2), 0.018762, 1e-5)
    close(bs.vega(100, 100, 1, 0.05, 0.2), 37.524, 1e-3)
    # finite-difference greeks
    h = 1e-3
    fd = (bs.price(100 + h, 100, 1, .05, .2) - bs.price(100 - h, 100, 1, .05, .2)) / (2 * h)
    close(fd, bs.delta(100, 100, 1, .05, .2), 1e-6)
    fd_t = -(bs.price(100, 100, 1 + h, .05, .2) - bs.price(100, 100, 1 - h, .05, .2)) / (2 * h)
    close(fd_t, bs.theta(100, 100, 1, .05, .2), 1e-4)
    # expiry
    assert bs.price(110, 100, 0, .05, .2) == 10.0
    assert bs.delta(90, 100, 0, .05, .2, "put") == -1.0


def test_implied_vol_roundtrip():
    for opt in ("call", "put"):
        p = bs.price(230, 250, 0.4, 0.04, 0.31, opt)
        close(bs.implied_vol(p, 230, 250, 0.4, 0.04, opt), 0.31, 1e-7)
    assert np.isnan(bs.implied_vol(0.0001, 100, 50, 1, 0.05, "call"))  # below lower bound


def test_black76_equals_bs():
    S, K, T, r, v = 100, 95, 0.7, 0.03, 0.25
    F = S * np.exp(r * T)
    for opt in ("call", "put"):
        close(bs.black76(F, K, T, r, v, opt), bs.price(S, K, T, r, v, opt), 1e-10)


def test_replicating_portfolio():
    for opt in ("call", "put"):
        rp = bs.replicating_portfolio(100, 105, 0.5, 0.04, 0.3, opt)
        close(rp["value"], bs.price(100, 105, 0.5, 0.04, 0.3, opt), 1e-10)


def test_parity_strategies_lock_profit():
    S, K, T, r = 100, 100, 1.0, 0.05
    C_fair = bs.price(S, K, T, r, 0.2, "call")
    P_fair = bs.price(S, K, T, r, 0.2, "put")
    res = arbitrage.parity_analysis(S, K, T, r, C_fair, P_fair)
    assert not res["arbitrage"]
    for C in (C_fair + 1.0, C_fair - 1.0):
        res = arbitrage.parity_analysis(S, K, T, r, C, P_fair, n=10)
        st = res["strategy"]
        close(st.net_cash_t0, 10 * abs(C - C_fair), 1e-9)
        grid = np.linspace(1, 300, 500)
        assert np.max(np.abs(st.payoff(grid))) < 1e-9


def test_bounds_strategies():
    S, K, T, r = 100, 90, 1.0, 0.05
    lo = S - K * np.exp(-r * T)
    out = arbitrage.bounds_analysis(S, K, T, r, C=lo - 2.0, P=K * np.exp(-r * T) + 1.0)
    assert len(out["strategies"]) == 2
    grid = np.linspace(1, 300, 500)
    for st in out["strategies"]:
        assert st.net_cash_t0 > 0
        assert st.payoff(grid).min() >= -1e-9


def test_model_strategy_cash():
    S, K, T, r, s = 100, 100, 0.5, 0.03, 0.2
    model = bs.price(S, K, T, r, s, "put")
    for mkt in (model + 0.8, model - 0.8):
        res = arbitrage.model_analysis(S, K, T, r, s, mkt, "put", n=5)
        close(res["strategy"].net_cash_t0, 5 * abs(mkt - model), 1e-9)


def test_portfolio_values():
    df = portfolio.default_positions()
    res, tot = portfolio.value_portfolio(df, 0.04)
    row = res.iloc[0]
    close(row["Market value (position)"], 100 * bs.price(230, 230, 0.25, 0.04, 0.25, "call"), 1e-8)
    assert res.iloc[1]["Market value (position)"] < 0  # short put
    close(tot["Market value"], res["Market value (position)"].sum(), 1e-9)


def test_hedging_quality():
    spec = hedging.HedgeSpec(K=100, T=0.5, r=0.03, sigma=0.2, option="call", position="Short", n=100,
                             K2=105, T2=0.75, option2="call", days_per_year=252)
    out = hedging.run_hedge([100, 101, 99.5, 102, 101.2, 103], spec)
    close(out["delta"]["Portfolio value"].iloc[0], 0.0, 1e-9)
    g = out["gamma"]
    assert np.max(np.abs(g["Net gamma"])) < 1e-9
    assert np.max(np.abs(g["Net delta"])) < 1e-9
    mc = hedging.mc_pnl(spec, 100, 0.03, 0.2, 5, 4000, seed=1)
    st = hedging.pnl_stats(mc).set_index("Strategy")
    assert st.loc["Delta hedge", "Std dev"] < 0.25 * st.loc["Unhedged", "Std dev"]
    assert st.loc["Delta-gamma hedge", "Std dev"] < st.loc["Delta hedge", "Std dev"]


def test_hagan_limits():
    # nu -> 0, beta = 1 : flat smile equal to alpha
    v = sabr.hagan_vol(100, np.array([80, 100, 120]), 1.0, 0.25, 1.0, 0.0, 1e-8)
    assert np.allclose(v, 0.25, atol=1e-7)
    # continuity at the money
    a = sabr.hagan_vol(100, 100.0, 1.0, 0.3, 0.5, -0.3, 0.6)
    b = sabr.hagan_vol(100, 100.0 + 1e-6, 1.0, 0.3, 0.5, -0.3, 0.6)
    close(a, b, 1e-7)
    # alpha_from_atm inverts the ATM formula
    al = sabr.alpha_from_atm(0.22, 100, 1.0, 0.5, -0.3, 0.6)
    close(sabr.atm_vol(100, 1.0, al, 0.5, -0.3, 0.6), 0.22, 1e-10)


def test_calibration_recovers_parameters():
    F, T = 105.0, 0.5
    K = np.linspace(70, 140, 15)
    true = dict(alpha=0.3, rho=-0.5, nu=0.9)
    vols = sabr.hagan_vol(F, K, T, true["alpha"], 1.0, true["rho"], true["nu"])
    cal = sabr.calibrate_smile(K, vols, F, T, beta=1.0)
    close(cal["alpha"], 0.3, 1e-5)
    close(cal["rho"], -0.5, 1e-4)
    close(cal["nu"], 0.9, 1e-4)
    # beta = 0.5
    al = 0.3 * F ** 0.5
    vols = sabr.hagan_vol(F, K, T, al, 0.5, 0.2, 0.5)
    cal = sabr.calibrate_smile(K, vols, F, T, beta=0.5)
    assert cal["rmse"] < 1e-6


def test_mc_matches_bs_when_no_volvol():
    S, K, T, r = 100.0, 105.0, 0.5, 0.03
    F0 = S * np.exp(r * T)
    sim = sabr_mc.simulate(F0, T, 0.2, 1.0, 0.0, 1e-8, 200_000, 50, "log", True, seed=3)
    p = sabr_mc.price_option(sim, K, r, "call")
    close(p["price"], bs.price(S, K, T, r, 0.2, "call"), 4 * p["std_error"] + 1e-3)
    u = sabr_mc.underlying_stats(sim, r)
    close(u["mean_ST"], F0, 4 * u["se_ST"] + 1e-6)


def test_mc_vs_hagan():
    S, r, T = 100.0, 0.02, 0.5
    F0 = S * np.exp(r * T)
    par = dict(alpha=0.25, beta=1.0, rho=-0.5, nu=0.6)
    sim = sabr_mc.simulate(F0, T, par["alpha"], 1.0, par["rho"], par["nu"], 200_000, 200, "log", True, seed=11)
    for K, opt in ((90, "put"), (100, "call"), (110, "call")):
        mc = sabr_mc.price_option(sim, K, r, opt)["price"]
        hg, _ = sabr.sabr_price(S, K, T, r, par["alpha"], 1.0, par["rho"], par["nu"], opt)
        close(mc, hg, 0.06)  # Hagan is an approximation; agreement within a few cents
    sim_e = sabr_mc.simulate(F0, T, par["alpha"], 1.0, par["rho"], par["nu"], 100_000, 200, "euler", True, seed=11)
    close(sabr_mc.price_option(sim_e, 100, r, "call")["price"],
          sabr_mc.price_option(sim, 100, r, "call")["price"], 0.08)


def test_historical_calibration():
    # simulate a long daily SABR path (beta = 1) and recover the parameters
    nus, rhos = [], []
    a, nu, rho, dt = 0.25, 0.8, -0.6, 1 / 252
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = 252 * 20
        z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
        al = np.exp(np.log(a) + np.concatenate([[0], np.cumsum(nu * np.sqrt(dt) * w2 - 0.5 * nu**2 * dt)])[:-1])
        S = 100 * np.exp(np.concatenate([[0], np.cumsum(al * np.sqrt(dt) * z1 - 0.5 * al**2 * dt)]))
        est = sabr.calibrate_historical(S, beta=1.0, block=21)
        assert est["alpha"] > 0
        nus.append(est["nu"])
        rhos.append(est["rho"])
    assert 0.6 < np.mean(nus) < 1.05, np.mean(nus)
    assert -0.8 < np.mean(rhos) < -0.35, np.mean(rhos)


def test_sample_file_roundtrip():
    df = sabr.make_sample_smile()
    parsed, info = sabr.parse_smile(df.drop(columns=["Implied Vol"]))
    assert np.allclose(parsed["IV"].values, df.sort_values("Strike")["Implied Vol"].values, atol=1e-4)


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        print("ok ", f.__name__)
    print(f"{len(fns)} tests passed")
