"""Shared Streamlit widgets: page header + Help dialog, live/manual spot input,
chart and table helpers, report downloads."""
from __future__ import annotations

import hashlib
import inspect
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import help_content, market

MODULE_NAMES = {
    "portfolio": "Portfolio Evaluation",
    "arbitrage": "Arbitrage Calculator",
    "hedging": "Delta & Delta-Gamma Hedging",
    "sabr": "SABR Calculator",
}
COLORS = ["#1f4e79", "#c0504d", "#2e8b57", "#d4a017", "#6a3d9a", "#7f7f7f"]


# ---------------------------------------------------------------------------
# Layout helpers (work on old and new Streamlit versions)
# ---------------------------------------------------------------------------
_WIDE_CACHE = {}


def _wide_kwargs(fn) -> dict:
    """New Streamlit: width="stretch"; older versions: use_container_width=True (checked from the signature)."""
    name = getattr(fn, "__name__", str(fn))
    if name not in _WIDE_CACHE:
        kw = {"use_container_width": True}
        try:
            params = inspect.signature(fn).parameters
            w = params.get("width")
            if w is not None and (isinstance(w.default, str) or "use_container_width" not in params):
                kw = {"width": "stretch"}
        except (TypeError, ValueError):
            pass
        _WIDE_CACHE[name] = kw
    return _WIDE_CACHE[name]


def _wide(fn, *args, **kwargs):
    return fn(*args, **_wide_kwargs(fn), **kwargs)


def show_fig(fig: go.Figure, key: str | None = None):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      font=dict(size=12), hovermode="x unified")
    cfg = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}
    kw = {"config": cfg}
    if key:
        kw["key"] = key
    return _wide(st.plotly_chart, fig, **kw)


def show_df(df: pd.DataFrame, fmt: dict | None = None, height: int | None = None, hide_index=True):
    kw = {"hide_index": hide_index}
    if height:
        kw["height"] = height
    data = df.style.format(fmt, na_rep="-") if fmt else df
    return _wide(st.dataframe, data, **kw)


def number_fmt(df: pd.DataFrame, default="{:,.4f}", overrides: dict | None = None) -> dict:
    fmt = {c: default for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    fmt.update(overrides or {})
    return fmt


def metric_row(items: list[tuple], cols: int | None = None):
    cols = cols or len(items)
    cs = st.columns(cols)
    for i, it in enumerate(items):
        label, value = it[0], it[1]
        delta = it[2] if len(it) > 2 else None
        cs[i % cols].metric(label, value, delta)


def money(x, dec=2):
    try:
        return f"{x:,.{dec}f}"
    except Exception:
        return str(x)


# ---------------------------------------------------------------------------
# Header + Help dialog
# ---------------------------------------------------------------------------
def _render_help(key: str):
    secs = help_content.get_help(key)
    titles = [s["title"] for s in secs]
    tabs = st.tabs(titles)
    for tab, sec in zip(tabs, secs):
        with tab:
            for kind, payload in sec["blocks"]:
                if kind == "md":
                    st.markdown(payload)
                else:
                    st.latex(help_content.latex_of(payload))
    st.caption("Formulas are generated with SymPy (sympy.latex).")


_DIALOGS = {}


def _dialog_for(key: str):
    if key not in _DIALOGS:
        title = f"Help - {MODULE_NAMES.get(key, key)}"
        if hasattr(st, "dialog"):
            _DIALOGS[key] = st.dialog(title, width="large")(lambda: _render_help(key))
        else:  # very old Streamlit: fall back to an inline expander
            _DIALOGS[key] = None
    return _DIALOGS[key]


def page_header(title: str, subtitle: str, help_key: str):
    c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
    with c1:
        st.title(title)
        st.caption(subtitle)
    with c2:
        clicked = st.button("❓ Help", key=f"help_btn_{help_key}", type="primary",
                            help="Theory, assumptions and formulas (SymPy) for this module")
    if clicked:
        dlg = _dialog_for(help_key)
        if dlg is not None:
            dlg()
        else:
            with st.expander("Help", expanded=True):
                _render_help(help_key)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_spot(ticker: str):
    return market.fetch_spot(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_history(ticker: str, period: str):
    return market.fetch_history(ticker, period)


def spot_input(prefix: str, default_ticker: str = "AAPL", default_spot: float = 100.0,
               label: str = "Spot price S") -> tuple[float, str, str]:
    """One field for the spot: 'Fetch live' fills it from Yahoo Finance, the user can always overwrite it."""
    kS, kT, kSrc = f"{prefix}_S", f"{prefix}_ticker", f"{prefix}_src"
    st.session_state.setdefault(kS, float(default_spot))
    st.session_state.setdefault(kT, default_ticker)
    st.session_state.setdefault(kSrc, "Manual input")

    def _fetch():
        try:
            px, desc = cached_spot(st.session_state[kT])
            st.session_state[kS] = round(float(px), 4)
            st.session_state[kSrc] = f"🟢 Live - {desc}"
        except Exception as exc:  # network / ticker errors -> keep manual value
            st.session_state[kSrc] = f"🟠 Manual input (live fetch failed: {exc})"

    def _manual():
        st.session_state[kSrc] = "✏️ Manual input"

    c1, c2 = st.columns([0.62, 0.38], vertical_alignment="bottom")
    c1.text_input("Ticker (Yahoo Finance)", key=kT)
    c2.button("🌐 Fetch live", key=f"{prefix}_fetch", on_click=_fetch)
    S = st.number_input(label, min_value=0.0001, step=1.0, format="%.4f", key=kS, on_change=_manual)
    st.caption(st.session_state[kSrc])
    return float(S), st.session_state[kT].strip().upper(), st.session_state[kSrc]


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def _signature(obj) -> str:
    try:
        raw = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        raw = str(obj)
    return hashlib.md5(raw.encode()).hexdigest()


def report_downloads(prefix: str, signature_obj, build_pdf, excel_sheets: dict | None = None,
                     file_stem: str = "report"):
    """PDF is built on demand (button) and cached for the current inputs; Excel is immediate."""
    sig = _signature(signature_obj)
    kp, ks = f"{prefix}_pdf", f"{prefix}_pdf_sig"
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📄 Build PDF report", key=f"{prefix}_buildpdf"):
            with st.spinner("Building PDF..."):
                try:
                    st.session_state[kp] = build_pdf()
                    st.session_state[ks] = sig
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")
    with c2:
        if st.session_state.get(ks) == sig and st.session_state.get(kp):
            st.download_button("⬇️ Download PDF", st.session_state[kp], file_name=f"{file_stem}.pdf",
                               mime="application/pdf", key=f"{prefix}_dlpdf")
        else:
            st.caption("Build the PDF first (printable tables + charts).")
    with c3:
        if excel_sheets:
            from core.reports import to_excel
            try:
                data = to_excel(excel_sheets)
                st.download_button("⬇️ Download Excel", data, file_name=f"{file_stem}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key=f"{prefix}_dlxlsx")
            except Exception as exc:
                st.caption(f"Excel export unavailable: {exc}")


def line_figure(x, series: dict, title: str, xlab: str = "", ylab: str = "", markers=False,
                hline=None, vline=None) -> go.Figure:
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=list(x), y=list(y), name=name, mode="lines+markers" if markers else "lines",
                                 line=dict(color=COLORS[i % len(COLORS)], width=2.2)))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#999")
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dot", line_color="#999")
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab, height=380)
    return fig
