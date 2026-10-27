"""Option Analytics Lab - entry point.

Run locally:   streamlit run app.py
Mobile:        deploy on Streamlit Community Cloud (see DEPLOY_MOBILE.md) and open the URL on Android / iPhone.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Option Analytics Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
      /* tighter, mobile-friendly spacing */
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
      @media (max-width: 640px) {
        .block-container {padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 3.2rem;}
        h1 {font-size: 1.6rem !important;}
      }
      div[data-testid="stMetricValue"] {font-size: 1.35rem;}
      .stTabs [data-baseweb="tab-list"] {flex-wrap: wrap;}
    </style>
    """,
    unsafe_allow_html=True,
)

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/portfolio.py", title="1 · Portfolio Evaluation", icon="💼"),
    st.Page("views/arbitrage.py", title="2 · Arbitrage Calculator", icon="⚖️"),
    st.Page("views/hedging.py", title="3 · Delta / Delta-Gamma Hedging", icon="🛡️"),
    st.Page("views/sabr.py", title="4 · SABR Calculator", icon="😊"),
]

with st.sidebar:
    st.markdown("### 📈 Option Analytics Lab")
    st.caption("European options · Black-Scholes · Arbitrage · Hedging · SABR")

pg = st.navigation(pages)
pg.run()

with st.sidebar:
    st.divider()
    st.caption("Each module has a ❓ **Help** button with the theory, assumptions and formulas.")
    st.caption("Market data: Yahoo Finance (yfinance). Every input can also be entered manually.")
