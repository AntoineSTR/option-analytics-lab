import streamlit as st

st.title("📈 Option Analytics Lab")
st.markdown(
    "A toolkit for **European options on non-dividend-paying stocks**: valuation, arbitrage detection, dynamic "
    "hedging and the **SABR** stochastic-volatility model. Every module accepts **live market data** (Yahoo Finance) "
    "or **manual inputs**, and has a **❓ Help** button explaining the theory, assumptions and formulas."
)

cards = [
    ("views/portfolio.py", "💼 1 · Portfolio Evaluation",
     "Market value, intrinsic value, time value and Greeks of a portfolio of calls and puts (long/short), "
     "with live spots, value profile charts and a printable report."),
    ("views/arbitrage.py", "⚖️ 2 · Arbitrage Calculator",
     "Synthetic options from stock + bonds, put-call parity, no-arbitrage bounds and Black-Scholes replication. "
     "Outputs the step-by-step strategy and proves it with the payoff at maturity."),
    ("views/hedging.py", "🛡️ 3 · Delta / Delta-Gamma Hedging",
     "Daily hedge adjustments on historical prices or on simulated spots (up to 5 days), P&L projection, "
     "Monte Carlo P&L distribution, charts and daily adjustment reports."),
    ("views/sabr.py", "😊 4 · SABR Calculator",
     "Hagan implied volatility, calibration from CSV/Excel (beta = 1), smile across spot/forward levels, "
     "SABR vs Black-Scholes pricing, mispricing strategies and Monte Carlo simulation of the SABR SDEs."),
]
c = st.columns(2)
for i, (page, title, desc) in enumerate(cards):
    with c[i % 2].container(border=True):
        st.markdown(f"#### {title}")
        st.write(desc)
        st.page_link(page, label="Open module →")

st.divider()
st.markdown("#### Conventions used everywhere")
st.markdown(
    "- **T** in years (3 months = 0.25, 30 days ≈ 30/365 = 0.0822).  \n"
    "- **r** and **σ** entered in **%** and used as continuously-compounded / annualised decimals.  \n"
    "- One option = right on **one share** (a listed contract usually covers 100 shares).  \n"
    "- Forward price of a non-dividend stock: **F = S·e^(rT)**.  \n"
    "- Charts: hover for values, use the camera icon to save a PNG. Each module builds a printable **PDF** "
    "report and an **Excel** export."
)
st.info("📱 On a phone, open the menu (☰ / ›) at the top-left to switch modules. Tables scroll horizontally.")
