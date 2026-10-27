import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import market, portfolio, reports
from ui import components as ui

ui.page_header("💼 Portfolio Evaluation",
               "Market value of a portfolio of European stock options - live or manual spot prices",
               "portfolio")

ss = st.session_state
ss.setdefault("pf_df", portfolio.default_positions())
ss.setdefault("pf_ver", 0)
ss.setdefault("pf_msgs", [])

c1, c2, c3 = st.columns([1, 1, 2])
r_pct = c1.number_input("Risk-free rate r (%)", value=4.00, step=0.25, format="%.2f", key="pf_r")
r = r_pct / 100.0
c2.metric("Positions", len(ss.pf_df))
c3.caption("Quantity = number of options, each on **one share** (1 listed contract = 100). "
           "Market price: optional option quote (else Black-Scholes). Premium: your entry price per option "
           "(optional, for P&L).")

st.subheader("Positions")
edited = ui._wide(
    st.data_editor,
    ss.pf_df,
    key=f"pf_editor_{ss.pf_ver}",
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Yahoo Finance symbol, e.g. AAPL, MSFT, MC.PA"),
        "Spot": st.column_config.NumberColumn("Spot", min_value=0.0, format="%.2f"),
        "Type": st.column_config.SelectboxColumn("Type", options=["Call", "Put"], required=True),
        "Position": st.column_config.SelectboxColumn("Position", options=["Long", "Short"], required=True),
        "Quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=1, format="%d"),
        "Strike": st.column_config.NumberColumn("Strike", min_value=0.0, format="%.2f"),
        "Maturity (years)": st.column_config.NumberColumn("Maturity (years)", min_value=0.0, step=0.01, format="%.4f"),
        "Volatility (%)": st.column_config.NumberColumn("Volatility (%)", min_value=0.0, step=0.5, format="%.2f"),
        "Market price": st.column_config.NumberColumn("Market price", min_value=0.0, format="%.4f",
                                                      help="Optional option quote. 0 = use Black-Scholes"),
        "Premium paid/received": st.column_config.NumberColumn("Premium paid/received", min_value=0.0, format="%.4f",
                                                               help="Optional entry price per option"),
    },
)

b1, b2, b3, _ = st.columns([1, 1, 1, 1])
fetch = b1.button("🌐 Fetch live spots", help="Download the last price of every ticker (Yahoo Finance)")
estvol = b2.button("📊 Estimate vols from history", help="1-year historical volatility of each ticker")
reset = b3.button("↺ Reset example")

if reset:
    ss.pf_df = portfolio.default_positions()
    ss.pf_ver += 1
    ss.pf_msgs = []
    st.rerun()

if fetch or estvol:
    df = edited.copy()
    msgs = []
    for tk in sorted({t for t in df["Ticker"].fillna("").astype(str).str.strip().str.upper() if t}):
        mask = df["Ticker"].fillna("").astype(str).str.strip().str.upper() == tk
        try:
            if fetch:
                px, desc = ui.cached_spot(tk)
                df.loc[mask, "Spot"] = round(px, 4)
                msgs.append(("ok", f"{tk}: spot {px:,.4f} ({desc})"))
            if estvol:
                hist = ui.cached_history(tk, "1y")
                vol = market.historical_volatility(hist)
                df.loc[mask, "Volatility (%)"] = round(vol * 100, 2)
                msgs.append(("ok", f"{tk}: 1y historical volatility {vol:.2%}"))
        except Exception as exc:
            msgs.append(("err", f"{tk}: {exc} - manual value kept"))
    ss.pf_df = df.reset_index(drop=True)
    ss.pf_ver += 1
    ss.pf_msgs = msgs
    st.rerun()

for kind, m in ss.pf_msgs:
    (st.success if kind == "ok" else st.warning)(m)

errs = portfolio.validate(edited)
if errs:
    for e in errs:
        st.error(e)
    st.stop()

res, tot = portfolio.value_portfolio(edited, r)
if res.empty:
    st.info("Add at least one position.")
    st.stop()

st.subheader("Portfolio value")
ui.metric_row([
    ("Market value", ui.money(tot["Market value"])),
    ("Intrinsic value", ui.money(tot["Intrinsic value"])),
    ("Time value", ui.money(tot["Time value"])),
    ("Unrealized P&L", ui.money(tot["Unrealized P&L"]) if tot["Cost basis"] != 0 else "n/a"),
])
ui.metric_row([
    ("Delta (shares)", ui.money(tot["Delta (shares)"])),
    ("Gamma", ui.money(tot["Gamma"], 4)),
    ("Vega (per vol pt)", ui.money(tot["Vega (per 1 vol pt)"])),
    ("Theta (per day)", ui.money(tot["Theta (per day)"])),
])
st.caption(f"Long positions worth {ui.money(tot['Long market value'])}; short positions "
           f"{ui.money(tot['Short market value'])} (liabilities). Values in the currency of the underlying.")

tab1, tab2, tab3 = st.tabs(["Positions detail", "By underlying", "Charts"])
with tab1:
    show_cols = ["Ticker", "Type", "Position", "Quantity", "Spot", "Strike", "T (y)", "Vol (%)", "Moneyness",
                 "Intrinsic / option", "BS value / option", "Time value / option", "Mark / option", "Valuation source",
                 "Intrinsic value (position)", "Market value (position)", "Unrealized P&L",
                 "Delta (shares)", "Gamma", "Vega (per 1 vol pt)", "Theta (per day)", "Rho (per 1% rate)"]
    ui.show_df(res[show_cols], ui.number_fmt(res[show_cols], "{:,.4f}",
                                              {"Quantity": "{:,.0f}", "Market value (position)": "{:,.2f}",
                                               "Intrinsic value (position)": "{:,.2f}", "Unrealized P&L": "{:,.2f}",
                                               "Spot": "{:,.2f}", "Strike": "{:,.2f}", "Vol (%)": "{:.2f}"}))
with tab2:
    byu = portfolio.by_underlying(res)
    ui.show_df(byu, ui.number_fmt(byu, "{:,.2f}", {"Gamma": "{:,.4f}"}))
with tab3:
    lab = [f"{a} {b} {c} K={k:g}" for a, b, c, k in res[["Position", "Ticker", "Type", "Strike"]].itertuples(index=False)]
    fig = go.Figure()
    fig.add_bar(x=lab, y=res["Intrinsic value (position)"], name="Intrinsic value", marker_color=ui.COLORS[0])
    fig.add_bar(x=lab, y=res["Market value (position)"] - res["Intrinsic value (position)"], name="Time value",
                marker_color=ui.COLORS[3])
    fig.update_layout(barmode="relative", title="Market value per position = intrinsic + time value", height=400)
    ui.show_fig(fig, key="pf_bar")

    tickers = list(dict.fromkeys(res["Ticker"]))
    tk = st.selectbox("Value profile for underlying", tickers, key="pf_prof_tk")
    prof = portfolio.value_profile(edited, r, tk)
    if not prof.empty:
        S0 = float(res.loc[res["Ticker"] == tk, "Spot"].iloc[0])
        fig2 = ui.line_figure(prof["Spot"], {"Value today (Black-Scholes)": prof["Value today (BS)"],
                                             "Intrinsic value / payoff": prof["Intrinsic value"]},
                              f"{tk} options: portfolio value vs spot", "Spot", "Value", hline=0, vline=S0)
        ui.show_fig(fig2, key="pf_prof")

st.subheader("Report")


def _build_pdf():
    rp = reports.PDFReport("Portfolio Evaluation Report", f"r = {r_pct:.2f}%")
    rp.heading("Portfolio totals")
    rp.kv({k: v for k, v in tot.items()})
    rp.heading("Inputs")
    rp.table(portfolio._clean(edited))
    rp.heading("Valuation per position")
    rp.table(res[["Ticker", "Type", "Position", "Quantity", "Spot", "Strike", "T (y)", "Vol (%)", "Moneyness",
                  "Intrinsic / option", "BS value / option", "Time value / option", "Mark / option",
                  "Intrinsic value (position)", "Market value (position)", "Unrealized P&L"]])
    rp.heading("Greeks per position")
    rp.table(res[["Ticker", "Type", "Position", "Quantity", "Delta (shares)", "Gamma", "Vega (per 1 vol pt)",
                  "Theta (per day)", "Rho (per 1% rate)"]])
    rp.heading("By underlying")
    rp.table(portfolio.by_underlying(res))
    for t in dict.fromkeys(res["Ticker"]):
        pr = portfolio.value_profile(edited, r, t)
        s0 = float(res.loc[res["Ticker"] == t, "Spot"].iloc[0])
        rp.figure(reports.line_fig(pr["Spot"], {"Value today (BS)": pr["Value today (BS)"],
                                                "Intrinsic value": pr["Intrinsic value"]},
                                   f"{t}: portfolio value vs spot", "Spot", "Value", hline=0, vline=s0))
    return rp.output()


ui.report_downloads("pf", {"df": edited.to_dict(), "r": r}, _build_pdf,
                    {"Inputs": portfolio._clean(edited), "Valuation": res, "By underlying": portfolio.by_underlying(res),
                     "Totals": tot}, "portfolio_report")
