import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import bs, hedging, reports
from core.sabr import parse_history, read_table
from ui import components as ui

ui.page_header("🛡️ Delta & Delta-Gamma Hedging",
               "Daily hedge adjustments, P&L projection and Spot Price Simulator (historical and forecast)",
               "hedging")
ss = st.session_state

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Option to hedge**")
        S, ticker, src = ui.spot_input("hed", "AAPL", 100.0)
        K = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0, key="hed_K")
    with c2:
        st.markdown("**Market parameters**")
        r = st.number_input("Risk-free rate r (%)", value=4.0, step=0.25, format="%.3f", key="hed_r") / 100
        sigma = st.number_input("Volatility σ (%)", min_value=0.5, value=25.0, step=0.5, format="%.2f", key="hed_sig") / 100
        T = st.number_input("Time to maturity T (years)", min_value=0.01, value=0.25, step=0.01, format="%.4f", key="hed_T")
        typ = st.selectbox("Option type", ["Call", "Put"], key="hed_type")
    with c3:
        st.markdown("**Book & hedge instrument**")
        pos = st.selectbox("Your position in this option", ["Short", "Long"], key="hed_pos",
                           help="A market maker who sold options is Short")
        n = st.number_input("Number of options", min_value=1.0, value=100.0, step=1.0, key="hed_n")
        typ2 = st.selectbox("Hedge option (delta-gamma) type", ["Call", "Put"], key="hed_type2")
        k2pct = st.number_input("Hedge option strike K2 (% of K)", min_value=10.0, value=105.0, step=1.0, key="hed_k2")
        T2 = st.number_input("Hedge option maturity T2 (years)", min_value=0.02, value=0.5, step=0.01, format="%.4f",
                             key="hed_T2")
    with st.expander("Advanced"):
        a1, a2 = st.columns(2)
        dpy = a1.number_input("Trading days per year", min_value=200, max_value=366, value=252, step=1, key="hed_dpy")
        tc = a2.number_input("Transaction cost (bps of traded notional)", min_value=0.0, value=0.0, step=1.0, key="hed_tc")

K2 = K * k2pct / 100


def make_spec(K_):
    return hedging.HedgeSpec(K=K_, T=T, r=r, sigma=sigma, option=typ.lower(), position=pos, n=n,
                             K2=K_ * k2pct / 100, T2=T2, option2=typ2.lower(), days_per_year=int(dpy), tc_bps=tc)


g0 = bs.greeks(S, K, T, r, sigma, typ)
g2 = bs.greeks(S, K2, T2, r, sigma, typ2)
q = (1 if pos == "Long" else -1) * n
w0 = -q * g0["gamma"] / g2["gamma"] if g2["gamma"] > 1e-12 else 0.0
st.markdown("**Initial hedge at the current spot**")
ui.metric_row([
    ("Option value", f"{g0['price']:,.4f}"), ("Delta", f"{g0['delta']:,.4f}"), ("Gamma", f"{g0['gamma']:,.5f}"),
    ("Delta hedge: shares", f"{-q * g0['delta']:+,.2f}"),
    ("Δ-Γ hedge: options 2", f"{w0:+,.2f}"),
    ("Δ-Γ hedge: shares", f"{-(q * g0['delta'] + w0 * g2['delta']):+,.2f}"),
])
st.caption(f"Hedge option: {typ2} K2 = {K2:,.2f}, T2 = {T2:.4f}y, value {g2['price']:,.4f}, delta {g2['delta']:.4f}, "
           f"gamma {g2['gamma']:.5f}.  Positive = buy, negative = sell.")


def render_results(spots, dates, spec, prefix, title):
    n_days = len(spots) - 1
    if spec.T <= n_days / spec.days_per_year:
        st.warning("The option expires inside the period: values after expiry are intrinsic values.")
    if spec.T2 <= n_days / spec.days_per_year:
        st.warning("The hedge option expires inside the period - choose a longer T2.")
    out = hedging.run_hedge(spots, spec, dates)
    sm = out["summary"]
    ui.metric_row([
        ("Spot move", f"{sm['Spot move (%)']:+.2f}%"),
        ("Unhedged P&L", ui.money(sm["Unhedged P&L"])),
        ("Delta-hedged P&L", ui.money(sm["Delta-hedged P&L"])),
        ("Delta-gamma P&L", ui.money(sm.get("Delta-gamma hedged P&L", np.nan))),
    ])
    x = out["delta"]["Date"].astype(str)
    series = {"Unhedged": out["unhedged"]["Cumulative P&L"], "Delta hedge": out["delta"]["Cumulative P&L"]}
    if out["gamma"] is not None:
        series["Delta-gamma hedge"] = out["gamma"]["Cumulative P&L"]
    cA, cB = st.columns(2)
    with cA:
        ui.show_fig(ui.line_figure(x, {"Spot": spots}, "Spot price path", "", "Spot", markers=True), key=f"{prefix}_spot")
    with cB:
        ui.show_fig(ui.line_figure(x, series, "Cumulative P&L", "", "P&L", markers=True, hline=0), key=f"{prefix}_pnl")
    hold = {"Shares (delta hedge)": out["delta"]["Shares held"]}
    if out["gamma"] is not None:
        hold["Shares (Δ-Γ hedge)"] = out["gamma"]["Shares held"]
        hold["Hedge options (Δ-Γ hedge)"] = out["gamma"]["Option 2 held"]
    ui.show_fig(ui.line_figure(x, hold, "Hedge holdings after each daily adjustment", "", "Units", markers=True,
                               hline=0), key=f"{prefix}_hold")

    st.markdown("##### Daily adjustment reports")
    t1, t2, t3 = st.tabs(["Delta hedge", "Delta-gamma hedge", "Unhedged"])
    with t1:
        ui.show_df(out["delta"], ui.number_fmt(out["delta"], "{:,.4f}"))
    with t2:
        if out["gamma"] is not None:
            ui.show_df(out["gamma"], ui.number_fmt(out["gamma"], "{:,.4f}"))
    with t3:
        ui.show_df(out["unhedged"], ui.number_fmt(out["unhedged"], "{:,.4f}"))

    def _pdf():
        rp = reports.PDFReport(f"Hedging Report - {title}", f"{typ} K={spec.K:,.2f}, {pos} {n:,.0f}")
        rp.heading("Parameters")
        rp.kv({"Option": f"{pos} {n:,.0f} {typ}", "Strike K": spec.K, "Maturity T (y)": spec.T, "Rate r": r,
               "Volatility": sigma, "Hedge option": f"{typ2} K2={spec.K2:,.2f} T2={spec.T2}",
               "Days per year": spec.days_per_year, "Transaction cost (bps)": tc, "Spot source": src})
        rp.heading("Summary")
        rp.kv(sm)
        rp.figure(reports.line_fig(x, {"Spot": spots}, "Spot path", markers=True))
        rp.figure(reports.line_fig(x, series, "Cumulative P&L", ylabel="P&L", markers=True, hline=0))
        rp.figure(reports.line_fig(x, hold, "Hedge holdings", ylabel="Units", markers=True, hline=0))
        rp.heading("Daily adjustment report - delta hedge")
        rp.table(out["delta"].drop(columns=["Unhedged cum. P&L"]))
        if out["gamma"] is not None:
            rp.heading("Daily adjustment report - delta-gamma hedge")
            rp.table(out["gamma"])
        rp.heading("Unhedged position")
        rp.table(out["unhedged"])
        return rp.output()

    ui.report_downloads(prefix, {"spots": list(map(float, spots)), "spec": spec.__dict__}, _pdf,
                        {"Summary": sm, "Delta hedge": out["delta"], "Delta-gamma hedge": out["gamma"],
                         "Unhedged": out["unhedged"]}, f"hedging_{prefix}")
    return out


tab_h, tab_f = st.tabs(["📜 Historical (backtest)", "🔮 Forecast - Spot Price Simulator"])

# ---------------------------------------------------------------- historical
with tab_h:
    st.caption("Replays the hedge on real daily closes: the option is set up on the first day of the window.")
    h1, h2, h3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    source = h1.radio("Price source", ["Yahoo Finance", "Upload CSV/Excel"], key="hed_src", horizontal=True)
    ndays = h2.slider("Number of past trading days", 2, 60, 10, key="hed_ndays")
    atm = h3.checkbox("Strike K = first close (ATM)", value=True, key="hed_atm",
                      help="Otherwise the strike K entered above is used")
    if source == "Yahoo Finance":
        if st.button(f"🌐 Download last {ndays} days of {ticker}", key="hed_dl"):
            try:
                ss.hed_hist = ui.cached_history(ticker, "6mo")
                ss.hed_hist_name = ticker
            except Exception as exc:
                st.error(f"{exc}. Upload a CSV instead (columns Date, Close).")
    else:
        up = st.file_uploader("CSV / Excel with columns Date, Close", type=["csv", "xlsx", "xls"], key="hed_up")
        if up is not None:
            try:
                ss.hed_hist = parse_history(read_table(up))
                ss.hed_hist_name = up.name
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
    hist = ss.get("hed_hist")
    if hist is not None and len(hist) >= 2:
        win = hist.iloc[-(ndays + 1):]
        st.caption(f"{ss.get('hed_hist_name')}: {win.index[0]:%Y-%m-%d} → {win.index[-1]:%Y-%m-%d} "
                   f"({len(win) - 1} re-balancing days)" if isinstance(win.index, pd.DatetimeIndex) else
                   f"{ss.get('hed_hist_name')}: {len(win) - 1} re-balancing days")
        K_h = float(round(win.iloc[0], 2)) if atm else K
        dates = [f"{d:%Y-%m-%d}" for d in win.index] if isinstance(win.index, pd.DatetimeIndex) else None
        render_results(win.values.astype(float), dates, make_spec(K_h), "hedh", "Historical")
    else:
        st.info("Download prices or upload a file to run the historical hedge.")

# ---------------------------------------------------------------- forecast
with tab_f:
    st.markdown("**Spot Price Simulator** - starts from the current spot S = "
                f"{S:,.4f} and projects up to 5 trading days.")
    f1, f2, f3 = st.columns(3)
    method = f1.radio("Generator", ["Random (GBM)", "Constant daily move", "Manual entry"], key="hed_method")
    fdays = f2.slider("Days to simulate", 1, 5, 5, key="hed_fdays")
    with f3:
        if method == "Random (GBM)":
            mu = st.number_input("Drift μ (% per year)", value=8.0, step=1.0, key="hed_mu") / 100
            sim_sig = st.number_input("Simulated volatility (%)", min_value=0.1, value=sigma * 100, step=1.0,
                                      key="hed_simsig") / 100
            seed = st.number_input("Random seed", min_value=0, value=7, step=1, key="hed_seed")
        elif method == "Constant daily move":
            move = st.number_input("Daily move (%)", value=1.0, step=0.25, key="hed_move") / 100
        else:
            st.caption("Type the spots in the table below.")
    if method == "Random (GBM)":
        path = hedging.simulate_gbm(S, mu, sim_sig, fdays, 1, int(seed), int(dpy))[0]
    elif method == "Constant daily move":
        path = S * (1 + move) ** np.arange(fdays + 1)
    else:
        path = np.full(fdays + 1, S)
    sig_key = (method, fdays, round(S, 6), tuple(np.round(path, 6)))
    if ss.get("hed_fsig") != sig_key:
        ss.hed_fsig = sig_key
        ss.hed_fver = ss.get("hed_fver", 0) + 1
        ss.hed_fdf = pd.DataFrame({"Day": [f"Day {i}" for i in range(fdays + 1)], "Spot": np.round(path, 4)})
    st.caption("You can edit any simulated spot (Day 0 = today's spot).")
    fdf = ui._wide(st.data_editor, ss.hed_fdf, key=f"hed_fed_{ss.hed_fver}", hide_index=True, disabled=["Day"],
                   column_config={"Spot": st.column_config.NumberColumn("Spot", min_value=0.0001, format="%.4f")})
    spots_f = pd.to_numeric(fdf["Spot"], errors="coerce").values
    if np.any(~np.isfinite(spots_f)) or np.any(spots_f <= 0):
        st.error("All spots must be positive numbers.")
    else:
        render_results(spots_f, list(fdf["Day"]), make_spec(K), "hedf", "Forecast")

    with st.expander("🎲 Monte Carlo: distribution of the projected P&L", expanded=False):
        m1, m2, m3 = st.columns(3)
        npaths = m1.number_input("Number of paths", 100, 50000, 5000, step=500, key="hed_np")
        mu_mc = m2.number_input("Drift μ (%)", value=8.0, step=1.0, key="hed_mumc") / 100
        sig_mc = m3.number_input("Realised volatility (%)", min_value=0.1, value=sigma * 100, step=1.0, key="hed_sigmc") / 100
        if st.button("Run Monte Carlo", key="hed_runmc"):
            mc = hedging.mc_pnl(make_spec(K), S, mu_mc, sig_mc, fdays, int(npaths), seed=123)
            ss.hed_mc = (mc, hedging.pnl_stats(mc))
        if ss.get("hed_mc"):
            mc, stats = ss.hed_mc
            ui.show_df(stats, ui.number_fmt(stats, "{:,.2f}"))
            fig = go.Figure()
            for i, col in enumerate([c for c in mc.columns if c != "Final spot"]):
                fig.add_trace(go.Histogram(x=mc[col], name=col, opacity=0.6, nbinsx=80,
                                           marker_color=ui.COLORS[i]))
            fig.update_layout(barmode="overlay", title=f"P&L after {fdays} day(s)", xaxis_title="P&L", height=380)
            ui.show_fig(fig, key="hed_mc_hist")
            st.caption("Hedging shrinks the P&L dispersion; delta-gamma hedging removes most of the remaining "
                       "convexity risk. If realised vol > σ, a short-gamma book loses on average (and vice versa).")
