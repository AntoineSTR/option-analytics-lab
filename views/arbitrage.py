import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import arbitrage, bs, hedging, reports
from ui import components as ui

ui.page_header("⚖️ Arbitrage Calculator",
               "Synthetic options from stock + bonds - detect mispricing and get the step-by-step strategy",
               "arbitrage")

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Underlying**")
        S, ticker, src = ui.spot_input("arb", "AAPL", 100.0)
    with c2:
        st.markdown("**Contract**")
        K = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0, key="arb_K")
        T = st.number_input("Maturity T (years)", min_value=0.0001, value=1.0, step=0.05, format="%.4f", key="arb_T")
        r = st.number_input("Risk-free rate r (%)", value=5.0, step=0.25, format="%.3f", key="arb_r") / 100
        n = st.number_input("Number of options (shares)", min_value=1.0, value=100.0, step=1.0, key="arb_n")
    with c3:
        st.markdown("**Market prices**")
        use_c = st.checkbox("Call quoted", True, key="arb_usec")
        C = st.number_input("Market call price C", min_value=0.0, value=12.00, step=0.05, format="%.4f",
                            key="arb_C", disabled=not use_c)
        use_p = st.checkbox("Put quoted", True, key="arb_usep")
        P = st.number_input("Market put price P", min_value=0.0, value=5.57, step=0.05, format="%.4f",
                            key="arb_P", disabled=not use_p)
        thr = st.number_input("Threshold per option (costs)", min_value=0.0, value=0.0, step=0.01, format="%.4f",
                              key="arb_thr", help="Gaps smaller than this are treated as no arbitrage")
    sig_pct = st.slider("Volatility σ for the model-based check (%)", 1.0, 150.0, 20.0, 0.5, key="arb_sig")
sig = sig_pct / 100
C_ = C if use_c else None
P_ = P if use_p else None
pvK = K * np.exp(-r * T)
st.caption(f"PV(K) = K·e^(-rT) = **{pvK:,.4f}** · Black-Scholes at σ = {sig_pct:.1f}%: "
           f"call {bs.price(S, K, T, r, sig, 'call'):,.4f}, put {bs.price(S, K, T, r, sig, 'put'):,.4f}")


def payoff_chart(strat, key):
    grid = np.linspace(0, 2.2 * K, 221)
    fig = go.Figure()
    for i, leg in enumerate(strat.legs, 1):
        fig.add_trace(go.Scatter(x=grid, y=leg.payoff(grid), name=f"Leg {i}: {leg.action.split(' (')[0]} {leg.kind}",
                                 line=dict(width=1.2, dash="dot")))
    net = strat.payoff(grid)
    fig.add_trace(go.Scatter(x=grid, y=net, name="Net payoff at T", line=dict(width=3, color="#1f4e79")))
    fig.add_trace(go.Scatter(x=grid, y=net + strat.net_cash_t0 * np.exp(r * T),
                             name="Total P&L at T (incl. cash invested)", line=dict(width=3, color="#2e8b57")))
    fig.add_hline(y=0, line_color="#999", line_dash="dash")
    fig.update_layout(title="Verification: payoff of every leg at maturity", xaxis_title="S_T",
                      yaxis_title="Payoff", height=420)
    ui.show_fig(fig, key=key)


def render_strategy(strat, key):
    t0 = strat.net_cash_t0
    st.markdown(f"##### {strat.title}")
    st.markdown(f"`{strat.verdict}`")
    st.markdown("**Execute today (t = 0):**")
    for s in strat.steps():
        st.markdown(f"- {s}")
    fv = t0 * np.exp(r * T)
    grid = np.linspace(0.01, 3 * K, 400)
    pay = strat.payoff(grid)
    ui.metric_row([("Net cash today", ui.money(t0)), ("Value at T (cash × e^rT)", ui.money(fv)),
                   ("Worst payoff at T", ui.money(pay.min())),
                   ("Type", "Riskless" if strat.model_free else "Model-dependent")])
    st.markdown(f"**At maturity:** {strat.maturity_note}")
    with st.expander("Legs and scenario table", expanded=False):
        ui.show_df(strat.legs_table(), {"Quantity": "{:,.4f}", "Unit price": "{:,.4f}", "Cash flow at t=0": "{:+,.2f}"})
        if strat.model_free:
            sc = strat.scenario_table(K, T, r)
            ui.show_df(sc, ui.number_fmt(sc, "{:,.2f}"))
    if strat.model_free:
        payoff_chart(strat, key)


results = {"inputs": {"S": S, "K": K, "T": T, "r": r, "C": C_, "P": P_, "n": n, "threshold": thr, "sigma": sig,
                      "spot source": src}}
tab1, tab2, tab3 = st.tabs(["1 · Put-call parity", "2 · No-arbitrage bounds", "3 · Black-Scholes replication"])

# ---------------------------------------------------------------- parity
with tab1:
    if C_ is None or P_ is None:
        st.info("Quote both a call and a put (same K and T) to use put-call parity.")
        par = None
    else:
        par = arbitrage.parity_analysis(S, K, T, r, C_, P_, n, thr)
        results["parity"] = par
        tbl = pd.DataFrame([
            {"Instrument": "Call", "Market price": C_, "Synthetic price": par["synthetic_call"],
             "Synthetic built with": "Long put + long stock + borrow PV(K)", "Market - synthetic": C_ - par["synthetic_call"]},
            {"Instrument": "Put", "Market price": P_, "Synthetic price": par["synthetic_put"],
             "Synthetic built with": "Long call + short stock + lend PV(K)", "Market - synthetic": P_ - par["synthetic_put"]},
        ])
        ui.show_df(tbl, {"Market price": "{:,.4f}", "Synthetic price": "{:,.4f}", "Market - synthetic": "{:+,.4f}"})
        if par["arbitrage"]:
            st.error(f"🚨 Arbitrage: parity gap of {par['gap']:+.4f} per option → "
                     f"{abs(par['gap']) * n:,.2f} locked today on {n:,.0f} options.")
            render_strategy(par["strategy"], "arb_par_chart")
        else:
            st.success(f"✅ No arbitrage: |C - P - S + PV(K)| = {abs(par['gap']):.4f} ≤ threshold {thr:.4f}.")

# ---------------------------------------------------------------- bounds
with tab2:
    bd = arbitrage.bounds_analysis(S, K, T, r, C_, P_, n, thr)
    results["bounds"] = bd
    if bd["bounds"]:
        bdf = pd.DataFrame(bd["bounds"])
        ui.show_df(bdf, {"Lower bound": "{:,.4f}", "Market price": "{:,.4f}", "Upper bound": "{:,.4f}"})
    if bd["strategies"]:
        for j, stg in enumerate(bd["strategies"]):
            st.error("🚨 Bound violated")
            render_strategy(stg, f"arb_bd_chart_{j}")
    else:
        st.success("✅ All quoted prices are within their model-free bounds.")

# ---------------------------------------------------------------- model
with tab3:
    st.caption("Compares each quoted option with its Black-Scholes replicating portfolio (Δ shares + bonds) at the "
               "volatility σ chosen above. Exploiting the gap requires dynamic re-balancing (module 3), so the profit "
               "is only expected, not riskless.")
    model_res = {}
    for opt, px in (("call", C_), ("put", P_)):
        if px is None:
            continue
        mr = arbitrage.model_analysis(S, K, T, r, sig, px, opt, n, thr)
        model_res[opt] = mr
        rep = mr["replication"]
        with st.container(border=True):
            st.markdown(f"#### {opt.capitalize()}")
            ui.metric_row([("Market price", f"{px:,.4f}"), ("Black-Scholes value", f"{mr['model_price']:,.4f}"),
                           ("Mispricing", f"{mr['mispricing']:+,.4f}"),
                           ("Implied vol", f"{mr['implied_vol']:.2%}" if np.isfinite(mr['implied_vol']) else "n/a")])
            st.markdown(f"Replicating portfolio per option: **{rep['shares']:+.4f} shares** "
                        f"({'buy' if rep['shares'] > 0 else 'short'}) and **{rep['bond']:+,.4f} in bonds** "
                        f"({'lend' if rep['bond'] > 0 else 'borrow'}) → value {rep['value']:,.4f}.")
            if mr["strategy"] is None:
                st.success("✅ Market price consistent with the model (within the threshold).")
                continue
            render_strategy(mr["strategy"], f"arb_model_{opt}")
            if st.checkbox("Simulate the dynamic replication until maturity (Monte Carlo, daily re-balancing)",
                           key=f"arb_sim_{opt}"):
                days = int(min(max(round(T * 252), 1), 756))
                pos = "Short" if mr["mispricing"] > 0 else "Long"
                spec = hedging.HedgeSpec(K=K, T=T, r=r, sigma=sig, option=opt, position=pos, n=n)
                mc = hedging.mc_pnl(spec, S, r, sig, days, 2000, seed=42)
                tot_pnl = mc["Delta hedge"] + abs(mr["mispricing"]) * n * np.exp(r * days / 252)
                fig = go.Figure(go.Histogram(x=tot_pnl, nbinsx=60, marker_color=ui.COLORS[2]))
                fig.add_vline(x=abs(mr["mispricing"]) * n * np.exp(r * T), line_dash="dash",
                              annotation_text="theoretical profit")
                fig.update_layout(title=f"P&L at T of the hedged trade ({days} daily re-balancings, 2,000 paths, "
                                        f"realised vol = {sig:.0%})", xaxis_title="P&L", height=380)
                ui.show_fig(fig, key=f"arb_mc_{opt}")
                st.caption(f"Mean {tot_pnl.mean():,.2f} · std {tot_pnl.std():,.2f} · P(loss) "
                           f"{(tot_pnl < 0).mean():.1%}. The dispersion comes from discrete re-balancing; it shrinks "
                           f"with more frequent hedging and grows if realised vol differs from σ.")
    results["model"] = model_res
    if not model_res:
        st.info("Quote at least one option price.")

st.subheader("Report")


def _build_pdf():
    rp = reports.PDFReport("Arbitrage Calculator Report", f"S = {S:,.4f} ({src})")
    rp.heading("Inputs")
    rp.kv({"Spot S": S, "Strike K": K, "Maturity T (y)": T, "Rate r": r, "PV(K)": pvK, "Call price": C_ or "-",
           "Put price": P_ or "-", "Options": n, "Threshold": thr, "Model sigma": sig})

    def add_strat(stg):
        rp.text(stg.title + "\n" + stg.verdict)
        rp.text("\n".join(stg.steps()))
        rp.table(stg.legs_table())
        rp.text(f"Net cash today: {stg.net_cash_t0:,.2f}   |   value at T: {stg.net_cash_t0 * np.exp(r * T):,.2f}")
        rp.text("At maturity: " + stg.maturity_note)
        if stg.model_free:
            rp.table(stg.scenario_table(K, T, r))
            grid = np.linspace(0, 2.2 * K, 221)
            series = {f"Leg {i}": l.payoff(grid) for i, l in enumerate(stg.legs, 1)}
            series["Net payoff"] = stg.payoff(grid)
            rp.figure(reports.line_fig(grid, series, "Payoff at maturity", "S_T", "Payoff", hline=0))

    rp.heading("1. Put-call parity")
    if par is None:
        rp.text("Not applicable (both call and put prices are required).")
    else:
        rp.kv({"Synthetic call P+S-PV(K)": par["synthetic_call"], "Synthetic put C-S+PV(K)": par["synthetic_put"],
               "Parity gap": par["gap"], "Arbitrage": par["arbitrage"]})
        if par["arbitrage"]:
            add_strat(par["strategy"])
    rp.heading("2. No-arbitrage bounds")
    rp.table(pd.DataFrame(bd["bounds"]))
    for stg in bd["strategies"]:
        add_strat(stg)
    rp.heading("3. Black-Scholes replication")
    for opt, mr in model_res.items():
        rp.kv({f"{opt} model value": mr["model_price"], "mispricing": mr["mispricing"],
               "implied vol": mr["implied_vol"], "replicating shares": mr["replication"]["shares"],
               "replicating bond": mr["replication"]["bond"]})
        if mr["strategy"] is not None:
            add_strat(mr["strategy"])
    return rp.output()


sheets = {"Inputs": results["inputs"]}
if par is not None and par["arbitrage"]:
    sheets["Parity strategy"] = par["strategy"].legs_table()
    sheets["Parity scenarios"] = par["strategy"].scenario_table(K, T, r)
if bd["bounds"]:
    sheets["Bounds"] = pd.DataFrame(bd["bounds"])
for opt, mr in model_res.items():
    if mr["strategy"] is not None:
        sheets[f"Model {opt} strategy"] = mr["strategy"].legs_table()
ui.report_downloads("arb", results["inputs"], _build_pdf, sheets, "arbitrage_report")
