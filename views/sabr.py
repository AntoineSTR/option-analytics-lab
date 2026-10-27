import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import arbitrage, bs, market, reports, sabr, sabr_mc
from ui import components as ui

ui.page_header("😊 SABR Calculator",
               "Hagan implied volatility, calibration (β = 1 for equities), smile, pricing, arbitrage and Monte Carlo",
               "sabr")
ss = st.session_state
for k, v in {"sabr_alpha": 0.27, "sabr_beta": 1.0, "sabr_rho": -0.55, "sabr_nu": 1.10,
             "sabr_T": 0.25, "sabr_r": 4.5}.items():
    ss.setdefault(k, v)

# ---------------------------------------------------------------- model inputs
with st.container(border=True):
    st.markdown("**Model inputs** (shared by every tab)")
    c1, c2, c3 = st.columns(3)
    with c1:
        S, ticker, src = ui.spot_input("sabr", "AAPL", 230.0)
    with c2:
        r = st.number_input("Risk-free rate r (%)", step=0.25, format="%.3f", key="sabr_r") / 100
        T = st.number_input("Maturity T (years)", min_value=0.005, step=0.05, format="%.4f", key="sabr_T")
        F = S * np.exp(r * T)
        st.metric("Forward F = S·e^(rT)", f"{F:,.4f}")
    with c3:
        p1, p2 = st.columns(2)
        alpha = p1.number_input("α (alpha)", min_value=1e-6, step=0.01, format="%.4f", key="sabr_alpha",
                                help="Volatility level (with β = 1 ≈ ATM vol)")
        beta = p2.number_input("β (beta)", min_value=0.0, max_value=1.0, step=0.1, format="%.2f", key="sabr_beta",
                               help="1 = lognormal (equities)")
        rho = p1.number_input("ρ (rho)", min_value=-0.999, max_value=0.999, step=0.05, format="%.3f", key="sabr_rho",
                              help="Spot/vol correlation (skew)")
        nu = p2.number_input("ν (nu)", min_value=0.0, step=0.05, format="%.4f", key="sabr_nu",
                             help="Vol of vol (smile curvature)")
        if ss.get("sabr_param_src"):
            st.caption(ss.sabr_param_src)

atm = sabr.atm_vol(F, T, alpha, beta, rho, nu)
tabs = st.tabs(["🧮 Calculator", "📂 Calibration", "😊 Smile & S/F grid", "💲 Pricing & arbitrage", "🎲 Monte Carlo"])

# ================================================================= 1. calculator
with tabs[0]:
    a1, a2, a3 = st.columns(3)
    ss.setdefault("sabr_K", float(round(S, 0)))
    K = a1.number_input("Strike K", min_value=0.01, step=1.0, key="sabr_K")
    otype = a2.selectbox("Option type", ["Call", "Put"], key="sabr_type")
    flat = a3.number_input("Flat Black-Scholes vol for comparison (%)", min_value=0.1, value=float(round(atm * 100, 2)),
                           step=0.5, key="sabr_flat") / 100
    px, iv = sabr.sabr_price(S, K, T, r, alpha, beta, rho, nu, otype)
    bs_px = bs.price(S, K, T, r, flat, otype)
    ui.metric_row([("Hagan implied vol σ_B", f"{iv:.4%}"), (f"{otype} price (SABR)", f"{px:,.4f}"),
                   (f"{otype} price (BS flat {flat:.2%})", f"{bs_px:,.4f}"), ("SABR − BS", f"{px - bs_px:+,.4f}")])
    ui.metric_row([("ATM vol (K = F)", f"{atm:.4%}"), ("K / F", f"{K / F:.4f}"),
                   ("SABR delta", f"{bs.black76_delta(F, K, T, r, iv, otype):.4f}"),
                   ("Intrinsic value", f"{bs.intrinsic(S, K, otype):,.4f}")])
    Ks = np.linspace(0.6 * F, 1.5 * F, 181)
    fig = ui.line_figure(Ks, {"SABR (Hagan)": sabr.hagan_vol(F, Ks, T, alpha, beta, rho, nu) * 100,
                              "Flat BS vol": np.full_like(Ks, flat * 100)},
                         "Implied volatility smile", "Strike K", "Implied vol (%)", vline=K)
    ui.show_fig(fig, key="sabr_calc_smile")

# ================================================================= 2. calibration
with tabs[1]:
    st.markdown("Upload **option quotes** (Strike + Implied Vol, or Market Price + Type) or **historical prices** "
                "(Date + Close). CSV or Excel. Files are recognised automatically.")
    sm = sabr.make_sample_smile()
    d1c, d2c, d3c = st.columns(3)
    d1c.download_button("⬇️ Sample option quotes (CSV)", sm.to_csv(index=False), "sample_sabr_option_quotes.csv",
                        "text/csv", key="sabr_s1")
    d2c.download_button("⬇️ Sample option quotes (Excel)", reports.to_excel({"quotes": sm}),
                        "sample_sabr_option_quotes.xlsx", key="sabr_s2")
    try:
        _sample = os.path.join(os.path.dirname(os.path.abspath(sabr.__file__)), "..", "data", "sample_historical_prices.csv")
        with open(_sample, "rb") as fh:
            d3c.download_button("⬇️ Sample historical prices (CSV)", fh.read(), "sample_historical_prices.csv",
                                "text/csv", key="sabr_s3")
    except OSError:
        pass

    src_mode = st.radio("Data source", ["Upload file", "Online option chain (Yahoo)", "Online price history (Yahoo)"],
                        horizontal=True, key="sabr_calsrc")
    beta_cal = st.number_input("β used for calibration (fixed)", 0.0, 1.0, 1.0, 0.1, key="sabr_betacal",
                               help="Equities: β = 1")
    data_kind, quotes, hist = None, None, None

    if src_mode == "Upload file":
        up = st.file_uploader("CSV / Excel file", type=["csv", "xlsx", "xls"], key="sabr_upload")
        if up is not None:
            try:
                raw = sabr.read_table(up)
                if sabr.is_history_table(raw):
                    data_kind, hist = "history", sabr.parse_history(raw)
                else:
                    quotes, info = sabr.parse_smile(raw, S=S, r=r, T=T)
                    data_kind = "quotes"
                    if info.get("iv_from_prices"):
                        st.info(f"{info['iv_from_prices']} implied vols computed from prices (Black-Scholes inversion).")
                    if info.get("iv_percent_converted"):
                        st.info("Implied vols given in % were converted to decimals.")
                st.caption(f"Loaded **{up.name}**: {len(raw)} rows, detected as "
                           f"**{'historical prices' if data_kind == 'history' else 'option quotes'}**.")
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
    elif src_mode == "Online option chain (Yahoo)":
        if st.button(f"🌐 List expiries for {ticker}", key="sabr_exps"):
            try:
                ss.sabr_expiries = market.fetch_option_expiries(ticker)
            except Exception as exc:
                st.error(str(exc))
        if ss.get("sabr_expiries"):
            exp = st.selectbox("Expiry", ss.sabr_expiries, key="sabr_exp")
            mny = st.slider("Strike range (% of spot)", 50, 150, (75, 125), key="sabr_mny")
            if st.button("Download chain", key="sabr_chain"):
                try:
                    calls, puts = market.fetch_option_chain(ticker, exp)
                    Texp = max((datetime.strptime(exp, "%Y-%m-%d").date() - date.today()).days, 1) / 365.0
                    otm = pd.concat([puts[puts.strike < S].assign(Type="Put"),
                                     calls[calls.strike >= S].assign(Type="Call")])
                    otm = otm[(otm["mid"] > 0) & (otm.strike >= S * mny[0] / 100) & (otm.strike <= S * mny[1] / 100)]
                    ss.sabr_chain_df = pd.DataFrame({"Strike": otm.strike, "Type": otm.Type, "Market Price": otm["mid"],
                                                     "Maturity (years)": Texp, "Spot": S})
                except Exception as exc:
                    st.error(str(exc))
        if ss.get("sabr_chain_df") is not None:
            quotes, _ = sabr.parse_smile(ss.sabr_chain_df, S=S, r=r)
            data_kind = "quotes"
            st.caption(f"{len(quotes)} out-of-the-money quotes (mid prices), implied vols computed by the app.")
    else:
        period = st.selectbox("History length", ["2y", "3y", "5y", "10y"], index=2, key="sabr_hper")
        if st.button(f"🌐 Download {period} of {ticker}", key="sabr_hdl"):
            try:
                ss.sabr_hist = ui.cached_history(ticker, period)
            except Exception as exc:
                st.error(str(exc))
        if ss.get("sabr_hist") is not None:
            data_kind, hist = "history", ss.sabr_hist

    cal = None
    if data_kind == "quotes" and quotes is not None and len(quotes):
        mats = sorted(quotes["T"].dropna().unique()) if quotes["T"].notna().any() else [T]
        Tsel = st.selectbox("Maturity to calibrate", mats, format_func=lambda v: f"{v:.4f} y", key="sabr_tsel") \
            if len(mats) > 1 else mats[0]
        qd = quotes[quotes["T"] == Tsel] if quotes["T"].notna().any() else quotes
        S_file = float(qd["Spot"].dropna().iloc[0]) if qd["Spot"].notna().any() else S
        F_cal = float(qd["Forward"].dropna().iloc[0]) if qd["Forward"].notna().any() else S_file * np.exp(r * Tsel)
        with st.expander("Data used", expanded=False):
            ui.show_df(qd, ui.number_fmt(qd, "{:,.4f}"))
        try:
            cal = sabr.calibrate_smile(qd["Strike"], qd["IV"], F_cal, float(Tsel), beta=beta_cal)
            cal["S"], cal["kind"] = S_file, "Option quotes (least squares)"
            cal["table"] = qd
        except Exception as exc:
            st.error(f"Calibration failed: {exc}")
    elif data_kind == "history" and hist is not None:
        hc1, hc2 = st.columns(2)
        block = hc1.select_slider("Block size m (days)", [5, 10, 21, 42, 63], value=21, key="sabr_block")
        aw = hc2.select_slider("Window for α (days)", [10, 21, 42, 63], value=21, key="sabr_aw")
        try:
            cal = sabr.calibrate_historical(hist, beta=beta_cal, block=block, alpha_window=aw)
            cal["kind"] = "Historical prices (statistical estimation)"
            cal["S"] = cal["S_last"]
            cal["hist"] = hist
            if not cal["noise_correction_ok"]:
                st.warning("Realised-vol sampling noise dominates: use a larger block or a longer history.")
        except Exception as exc:
            st.error(str(exc))

    if cal:
        ss.sabr_cal = cal
        st.success(f"Calibrated ({cal['kind']}): α = {cal['alpha']:.4f}, β = {cal['beta']:.2f}, "
                   f"ρ = {cal['rho']:.4f}, ν = {cal['nu']:.4f}")
        if "rmse" in cal:
            ui.metric_row([("α", f"{cal['alpha']:.4f}"), ("ρ", f"{cal['rho']:.4f}"), ("ν", f"{cal['nu']:.4f}"),
                           ("RMSE (vol pts)", f"{cal['rmse'] * 100:.3f}"), ("Max error (vol pts)",
                                                                           f"{cal['max_abs_error'] * 100:.3f}")])
            Kf = np.linspace(cal["strikes"].min() * 0.95, cal["strikes"].max() * 1.05, 200)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cal["strikes"], y=cal["market_vols"] * 100, mode="markers", name="Market",
                                     marker=dict(size=8, color=ui.COLORS[1])))
            fig.add_trace(go.Scatter(x=Kf, y=sabr.hagan_vol(cal["F"], Kf, cal["T"], cal["alpha"], cal["beta"],
                                                            cal["rho"], cal["nu"]) * 100,
                                     name="SABR fit", line=dict(color=ui.COLORS[0], width=2.5)))
            fig.add_vline(x=cal["F"], line_dash="dot", annotation_text="F")
            fig.update_layout(title="Calibrated smile vs market", xaxis_title="Strike", yaxis_title="Implied vol (%)",
                              height=400)
            ui.show_fig(fig, key="sabr_calfit")
            res_df = pd.DataFrame({"Strike": cal["strikes"], "Market vol": cal["market_vols"],
                                   "SABR vol": cal["fitted_vols"], "Error (vol pts)": (cal["fitted_vols"] - cal["market_vols"]) * 100})
            ui.show_df(res_df, {"Strike": "{:,.2f}", "Market vol": "{:.4%}", "SABR vol": "{:.4%}",
                                "Error (vol pts)": "{:+.3f}"})
        else:
            ui.metric_row([("α (current vol)", f"{cal['alpha']:.4f}"), ("ρ", f"{cal['rho']:.4f}"),
                           ("ν", f"{cal['nu']:.4f}"), ("Blocks used", cal["n_blocks"]),
                           ("Mean realised vol", f"{cal['mean_realised_vol']:.2%}")])
            st.caption(f"Raw (uncorrected) estimates: ρ = {cal['rho_uncorrected']:.3f}, ν = {cal['nu_uncorrected']:.3f}. "
                       "The app corrects for realised-vol sampling noise and block averaging (see Help).")
            det = cal["details"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=det["Block end"], y=det["Realised vol (ann.)"] * 100, name="Realised vol (%)",
                                     line=dict(color=ui.COLORS[0])))
            fig.update_layout(title="Realised volatility per block (the vol of this series gives ν)", height=360,
                              yaxis_title="%")
            ui.show_fig(fig, key="sabr_hvol")

        def _apply():
            c = ss.sabr_cal
            ss.sabr_alpha, ss.sabr_beta = float(c["alpha"]), float(c["beta"])
            ss.sabr_rho, ss.sabr_nu = float(np.clip(c["rho"], -0.999, 0.999)), float(c["nu"])
            if "T" in c:
                ss.sabr_T = float(c["T"])
            if c.get("S"):
                s_new = float(c["S"])
                ss.sabr_S = s_new
                ss.sabr_src = "📂 Spot from calibration data"
                # re-centre strikes and comparison vols on the new spot / smile
                ss.sabr_K = ss.sabr_Kmc = float(round(s_new, 0))
                ss.sabr_mK = float(round(s_new * 1.05, 0))
                for k_ in ("sabr_mpx", "sabr_flat", "sabr_flat2"):
                    ss.pop(k_, None)
            ss.sabr_param_src = f"Parameters from calibration: {c['kind']}"

        st.button("✅ Use these parameters in all tabs", on_click=_apply, key="sabr_apply", type="primary")

# ================================================================= 3. smile & grid
with tabs[2]:
    g1, g2 = st.columns(2)
    shifts = g1.multiselect("Spot levels (% change vs current S)", [-30, -20, -10, -5, 0, 5, 10, 20, 30],
                            default=[-20, -10, 0, 10, 20], key="sabr_shifts")
    krange = g2.slider("Strike range (% of current S)", 40, 200, (60, 150), key="sabr_krange")
    nK = st.slider("Number of strikes in the table", 5, 25, 11, key="sabr_nk")
    spots = sorted(S * (1 + np.array(shifts or [0]) / 100))
    Kc = np.linspace(S * krange[0] / 100, S * krange[1] / 100, 200)
    fig = go.Figure()
    for i, s_ in enumerate(spots):
        F_ = s_ * np.exp(r * T)
        fig.add_trace(go.Scatter(x=Kc, y=sabr.hagan_vol(F_, Kc, T, alpha, beta, rho, nu) * 100,
                                 name=f"S = {s_:,.2f} (F = {F_:,.2f})", line=dict(width=2.2, color=ui.COLORS[i % 6])))
    fig.update_layout(title="SABR smile for several spot / forward levels", xaxis_title="Strike K",
                      yaxis_title="Implied vol (%)", height=420)
    ui.show_fig(fig, key="sabr_multi")
    st.caption("With β = 1 the smile moves with the forward (sticky-delta): the ATM vol follows the backbone "
               "σ_ATM(F) while ρ sets the skew and ν the curvature.")
    Kt = np.round(np.linspace(S * krange[0] / 100, S * krange[1] / 100, nK), 2)
    grid = sabr.smile_grid(alpha, beta, rho, nu, spots, Kt, r, T)
    hm = go.Figure(go.Heatmap(z=grid.drop(columns="Forward F").values * 100, x=[f"{k:g}" for k in Kt],
                              y=[f"S={s_:,.1f} | F={f_:,.1f}" for s_, f_ in zip(grid.index, grid["Forward F"])],
                              colorscale="Viridis", colorbar=dict(title="IV %")))
    hm.update_layout(title="Implied vol across (S, F) and K", xaxis_title="Strike", height=380)
    ui.show_fig(hm, key="sabr_heat")
    gshow = grid.copy()
    gshow.insert(0, "Spot S", gshow.index)
    gshow.columns = [str(c) for c in gshow.columns]
    ui.show_df(gshow, {c: ("{:,.2f}" if c in ("Spot S", "Forward F") else "{:.2%}") for c in gshow.columns})

# ================================================================= 4. pricing & arbitrage
with tabs[3]:
    cal = ss.get("sabr_cal")
    use_mkt = cal is not None and "table" in cal and st.checkbox(
        "Use the strikes and market prices of the calibration data", value=True, key="sabr_usemkt")
    pt1, pt2 = st.columns(2)
    ptype = pt1.selectbox("Option type for the table", ["Call", "Put"], key="sabr_ptype")
    flat2 = pt2.number_input("Flat BS vol (%)", min_value=0.1, value=float(round(atm * 100, 2)), step=0.5,
                             key="sabr_flat2") / 100
    if use_mkt:
        qd = cal["table"]
        qd = qd[qd["Type"] == ptype]
        strikes = qd["Strike"].values
        mkt = qd["Price"].values if qd["Price"].notna().any() else None
        if len(strikes) == 0:
            st.info(f"No {ptype.lower()} quotes in the data: showing a strike grid instead.")
            strikes, mkt = np.round(np.linspace(0.8 * S, 1.2 * S, 9), 2), None
    else:
        strikes, mkt = np.round(np.linspace(0.8 * S, 1.2 * S, 9), 2), None
    ptab = sabr.pricing_table(S, r, T, alpha, beta, rho, nu, strikes, ptype.lower(), flat2, mkt)
    st.markdown(f"**Prices, forward, Black-Scholes and SABR values** (S = {S:,.4f}, F = {F:,.4f}, T = {T:.4f})")
    ui.show_df(ptab, ui.number_fmt(ptab, "{:,.4f}", {"SABR implied vol": "{:.4%}", "Flat BS vol": "{:.4%}"}))
    figp = ui.line_figure(ptab["Strike"], {"SABR (Hagan)": ptab.iloc[:, 6], "Black-Scholes flat vol": ptab.iloc[:, 8]},
                          f"{ptype} prices: SABR vs Black-Scholes", "Strike", "Price", markers=True)
    if mkt is not None:
        figp.add_trace(go.Scatter(x=ptab["Strike"], y=ptab["Market price"], mode="markers", name="Market",
                                  marker=dict(size=9, color=ui.COLORS[1], symbol="x")))
    ui.show_fig(figp, key="sabr_pricefig")

    st.markdown("---")
    st.markdown("#### Mispriced option → arbitrage strategy")
    m1, m2, m3, m4 = st.columns(4)
    ss.setdefault("sabr_mK", float(round(S * 1.05, 0)))
    Km = m1.number_input("Strike", min_value=0.01, step=1.0, key="sabr_mK")
    tm = m2.selectbox("Type", ["Call", "Put"], key="sabr_mtype")
    fair, ivm = sabr.sabr_price(S, Km, T, r, alpha, beta, rho, nu, tm)
    pm = m3.number_input("Market price", min_value=0.0, value=float(round(fair * 1.08, 2)), step=0.05,
                         format="%.4f", key="sabr_mpx")
    nm = m4.number_input("Quantity", min_value=1.0, value=100.0, step=1.0, key="sabr_mn")
    mr = arbitrage.model_analysis(S, Km, T, r, ivm, pm, tm.lower(), nm, 0.0)
    ui.metric_row([("SABR price", f"{fair:,.4f}"), ("SABR implied vol", f"{ivm:.4%}"),
                   ("Market implied vol", f"{mr['implied_vol']:.4%}" if np.isfinite(mr["implied_vol"]) else "n/a"),
                   ("Mispricing / option", f"{pm - fair:+,.4f}")])
    if mr["strategy"] is None:
        st.success("Market price equals the SABR price: no trade.")
    else:
        stg = mr["strategy"]
        st.error(f"🚨 {stg.title}")
        for s_ in stg.steps():
            st.markdown(f"- {s_}")
        st.markdown(f"**Expected profit:** {abs(pm - fair) * nm:,.2f} today (≈ {abs(pm - fair) * nm * np.exp(r * T):,.2f} "
                    f"at T). {stg.maturity_note.replace('model volatility', 'SABR volatility')}")
        st.caption("Hedge ratio = Black-Scholes delta evaluated at the SABR implied vol of this strike. A model-free "
                   "alternative is to trade against put-call parity (module 2) when the other option is quoted.")

    st.markdown("---")
    st.markdown("#### Static arbitrage check of the SABR smile (butterfly / density)")
    dens = sabr.density_check(S, r, T, alpha, beta, rho, nu)
    neg = dens[dens["Density"] < -1e-10]
    figd = ui.line_figure(dens["Strike"], {"Risk-neutral density q(K)": dens["Density"]},
                          "Breeden-Litzenberger density implied by the Hagan smile", "Strike", "q(K)", hline=0)
    ui.show_fig(figd, key="sabr_dens")
    if neg.empty:
        st.success("✅ Density non-negative on [0.3F, 2F]: no butterfly arbitrage in the smile.")
    else:
        st.warning(f"⚠️ Negative density for strikes between {neg['Strike'].min():,.2f} and {neg['Strike'].max():,.2f}: "
                   "the Hagan approximation allows a butterfly arbitrage there (buy the butterfly, which has a "
                   "negative price, and receive a non-negative payoff).")

# ================================================================= 5. Monte Carlo
with tabs[4]:
    st.markdown("Simulates the SABR SDEs **dF = αF^β dW₁, dα = ναdW₂, dW₁dW₂ = ρdt** to compute the final price of the "
                "underlying and the present value of the average option payoff.")
    mc1, mc2, mc3 = st.columns(3)
    scheme = mc1.radio("Scheme", list(sabr_mc.SCHEMES.keys()), format_func=lambda k: sabr_mc.SCHEMES[k],
                       index=1, key="sabr_scheme")
    nsim = mc2.number_input("Number of simulations", min_value=100, max_value=2_000_000, value=50_000, step=10_000,
                            key="sabr_nsim")
    nsteps = mc2.number_input("Time steps per path", min_value=1, max_value=2000, value=int(max(1, round(T * 252))),
                              step=10, key="sabr_nsteps", help="Default: one step per trading day")
    anti = mc3.checkbox("Antithetic variates", value=True, key="sabr_anti")
    seed = mc3.number_input("Random seed", min_value=0, value=2026, step=1, key="sabr_seed")
    o1, o2, o3 = st.columns(3)
    ss.setdefault("sabr_Kmc", float(round(S, 0)))
    Kmc = o1.number_input("Option strike", min_value=0.01, step=1.0, key="sabr_Kmc")
    tmc = o2.selectbox("Option type", ["Call", "Put"], key="sabr_tmc")
    compare = o3.checkbox("Also run the other scheme (comparison)", value=False, key="sabr_cmp")
    if nsim * nsteps > 2e8:
        st.warning("Very large run (simulations × steps > 200 million): it may take a while, especially online.")
    if st.button("▶️ Run simulation", type="primary", key="sabr_run"):
        with st.spinner("Simulating..."):
            runs = {}
            for sch in ([scheme, "euler" if scheme == "log" else "log"] if compare else [scheme]):
                sim = sabr_mc.simulate(F, T, alpha, beta, rho, nu, int(nsim), int(nsteps), sch, anti, int(seed))
                runs[sch] = (sim, sabr_mc.price_option(sim, Kmc, r, tmc), sabr_mc.underlying_stats(sim, r))
            ss.sabr_mc = {"runs": runs, "K": Kmc, "type": tmc, "params": (S, r, T, alpha, beta, rho, nu)}
    res = ss.get("sabr_mc")
    if res:
        S0_, r_, T_, a_, b_, rh_, nu_ = res["params"]
        F_ = S0_ * np.exp(r_ * T_)
        hag, ivh = sabr.sabr_price(S0_, res["K"], T_, r_, a_, b_, rh_, nu_, res["type"])
        sch0 = list(res["runs"].keys())[0]
        sim, pr, us = res["runs"][sch0]
        st.markdown(f"**{sabr_mc.SCHEMES[sch0]}** - {sim['n_paths']:,} paths × {sim['n_steps']} steps "
                    f"(Δt = {sim['dt']:.5f} y)")
        ui.metric_row([("Expected final price E[S_T]", f"{us['mean_ST']:,.4f}", f"F₀ = {F_:,.4f}"),
                       ("PV of E[S_T]", f"{us['pv_mean_ST']:,.4f}", f"S₀ = {S0_:,.4f}"),
                       (f"{res['type']} price (MC)", f"{pr['price']:,.4f}", f"± {1.96 * pr['std_error']:,.4f} (95%)"),
                       ("Hagan price", f"{hag:,.4f}", f"MC − Hagan {pr['price'] - hag:+.4f}")])
        st.caption(f"Mean payoff (undiscounted) {pr['mean_payoff']:,.4f} → PV = e^(-rT) × mean = {pr['price']:,.4f}. "
                   f"95% CI [{pr['ci_low']:,.4f}, {pr['ci_high']:,.4f}] · P(in the money) {pr['itm_prob']:.1%} · "
                   f"martingale check E[S_T] − F₀ = {us['martingale_error']:+.4f} (± {1.96 * us['se_ST']:.4f}) · "
                   f"paths with F < 0: {sim['pct_negative_F']:.2f}%, α < 0: {sim['pct_negative_alpha']:.2f}%")
        if len(res["runs"]) > 1:
            rows = [{"Scheme": sabr_mc.SCHEMES[k], "Price": v[1]["price"], "Std error": v[1]["std_error"],
                     "E[S_T]": v[2]["mean_ST"], "E[S_T] − F₀": v[2]["martingale_error"],
                     "% paths F<0": v[0]["pct_negative_F"], "% paths α<0": v[0]["pct_negative_alpha"]}
                    for k, v in res["runs"].items()]
            rows.append({"Scheme": "Hagan closed form", "Price": hag})
            ui.show_df(pd.DataFrame(rows), {"Price": "{:,.4f}", "Std error": "{:,.4f}", "E[S_T]": "{:,.4f}",
                                            "E[S_T] − F₀": "{:+,.4f}", "% paths F<0": "{:.2f}", "% paths α<0": "{:.2f}"})
        cc1, cc2 = st.columns(2)
        with cc1:
            run = pr["running_mean"]
            mult = 2 if sim["antithetic"] else 1
            idx = np.unique(np.geomspace(1, run.size, 400).astype(int)) - 1
            figc = ui.line_figure((idx + 1) * mult, {"MC estimate": run[idx], "Hagan": np.full(idx.size, hag)},
                                  "Convergence of the MC price", "Number of paths", "Price")
            figc.update_xaxes(type="log")
            ui.show_fig(figc, key="sabr_conv")
        with cc2:
            fh = go.Figure(go.Histogram(x=sim["FT"], nbinsx=100, marker_color=ui.COLORS[0]))
            fh.add_vline(x=res["K"], line_dash="dash", annotation_text="K")
            fh.update_layout(title="Distribution of the final price S_T = F_T", xaxis_title="S_T", height=380)
            ui.show_fig(fh, key="sabr_hist")
        p1c, p2c = st.columns(2)
        with p1c:
            fpth = go.Figure()
            for j in range(min(20, sim["paths_F"].shape[0])):
                fpth.add_trace(go.Scatter(x=sim["time"], y=sim["paths_F"][j], mode="lines", line=dict(width=1),
                                          showlegend=False))
            fpth.update_layout(title="Sample paths of F_t", xaxis_title="t (years)", height=360, hovermode=False)
            ui.show_fig(fpth, key="sabr_pf")
        with p2c:
            fpa = go.Figure()
            for j in range(min(20, sim["paths_alpha"].shape[0])):
                fpa.add_trace(go.Scatter(x=sim["time"], y=sim["paths_alpha"][j], mode="lines", line=dict(width=1),
                                         showlegend=False))
            fpa.update_layout(title="Sample paths of α_t (stochastic vol)", xaxis_title="t (years)", height=360,
                              hovermode=False)
            ui.show_fig(fpa, key="sabr_pa")
        conv = sabr_mc.convergence_table(sim, res["K"], r_, res["type"], hag)
        if st.checkbox("Monte Carlo smile vs Hagan (same paths)", key="sabr_mcsmile"):
            Ksm = np.round(np.linspace(0.75 * S0_, 1.3 * S0_, 12), 2)
            mcs = sabr_mc.mc_smile(sim, Ksm, r_, S0_)
            mcs["Hagan vol"] = sabr.hagan_vol(F_, Ksm, T_, a_, b_, rh_, nu_)
            fs = go.Figure()
            fs.add_trace(go.Scatter(x=mcs["Strike"], y=mcs["MC implied vol"] * 100, mode="markers+lines", name="MC"))
            fs.add_trace(go.Scatter(x=mcs["Strike"], y=mcs["Hagan vol"] * 100, name="Hagan"))
            fs.update_layout(title="Implied vol: Monte Carlo vs Hagan's approximation", xaxis_title="Strike",
                             yaxis_title="%", height=380)
            ui.show_fig(fs, key="sabr_mcsm")
            ui.show_df(mcs, {"Strike": "{:,.2f}", "MC price": "{:,.4f}", "Std error": "{:,.4f}",
                             "MC implied vol": "{:.4%}", "Hagan vol": "{:.4%}"})

# ================================================================= report
st.subheader("Report")


def _build_pdf():
    rp = reports.PDFReport("SABR Calculator Report", f"S = {S:,.4f} ({src})")
    rp.heading("Model inputs")
    rp.kv({"Spot S": S, "Rate r": r, "Maturity T": T, "Forward F": F, "alpha": alpha, "beta": beta, "rho": rho,
           "nu": nu, "ATM vol": atm, "Parameter source": ss.get("sabr_param_src", "manual")})
    c = ss.get("sabr_cal")
    if c:
        rp.heading("Calibration")
        rp.kv({k: c[k] for k in ("kind", "alpha", "beta", "rho", "nu") if k in c} |
              ({"RMSE (vol)": c["rmse"], "Max error (vol)": c["max_abs_error"]} if "rmse" in c else
               {"blocks": c["n_blocks"], "rho uncorrected": c["rho_uncorrected"], "nu uncorrected": c["nu_uncorrected"]}))
        if "rmse" in c:
            Kf = np.linspace(c["strikes"].min(), c["strikes"].max(), 200)
            rp.figure(reports.scatter_line_fig(c["strikes"], c["market_vols"], Kf,
                                               sabr.hagan_vol(c["F"], Kf, c["T"], c["alpha"], c["beta"], c["rho"], c["nu"]),
                                               "Calibrated smile"))
    rp.heading("Pricing table")
    rp.table(ptab)
    Ks_ = np.linspace(0.6 * F, 1.5 * F, 200)
    rp.figure(reports.line_fig(Ks_, {f"S={s_:,.1f}": sabr.hagan_vol(s_ * np.exp(r * T), Ks_, T, alpha, beta, rho, nu)
                                     for s_ in spots}, "SABR smiles for several spot levels", "Strike", "Implied vol"))
    rp.heading("Implied vol grid (rows: spot / forward, columns: strikes)")
    rp.table(gshow)
    rp.heading("Mispriced option")
    rp.kv({"Strike": Km, "Type": tm, "Market price": pm, "SABR price": fair, "SABR vol": ivm,
           "Mispricing": pm - fair})
    if mr["strategy"] is not None:
        rp.text("\n".join(mr["strategy"].steps()))
    m = ss.get("sabr_mc")
    if m:
        rp.heading("Monte Carlo")
        for k, (sim_, pr_, us_) in m["runs"].items():
            rp.kv({"Scheme": sabr_mc.SCHEMES[k], "Paths": sim_["n_paths"], "Steps": sim_["n_steps"],
                   "Strike": m["K"], "Type": m["type"], "E[S_T]": us_["mean_ST"], "PV E[S_T]": us_["pv_mean_ST"],
                   "MC price": pr_["price"], "Std error": pr_["std_error"], "CI low": pr_["ci_low"],
                   "CI high": pr_["ci_high"]})
        sim_, pr_, _ = list(m["runs"].values())[0]
        rp.figure(reports.hist_fig({"S_T": sim_["FT"]}, "Distribution of S_T", "S_T"))
        rp.table(sabr_mc.convergence_table(sim_, m["K"], m["params"][1], m["type"]))
    return rp.output()


sheets = {"Inputs": {"S": S, "r": r, "T": T, "F": F, "alpha": alpha, "beta": beta, "rho": rho, "nu": nu},
          "Pricing": ptab, "IV grid": gshow}
if ss.get("sabr_cal") and "table" in ss.sabr_cal:
    sheets["Calibration data"] = ss.sabr_cal["table"]
ui.report_downloads("sabr", {"p": [S, r, T, alpha, beta, rho, nu], "mc": bool(ss.get("sabr_mc")),
                             "cal": str(ss.get("sabr_param_src"))}, _build_pdf, sheets, "sabr_report")
