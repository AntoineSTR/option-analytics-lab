"""Headless smoke test: executes every page with a fake Streamlit, offline market data,
pressing buttons, uploading the sample files and building every PDF/Excel report."""
import io
import os
import runpy
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import fake_streamlit  # noqa: E402

st = fake_streamlit.install()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import market  # noqa: E402


def fake_spot(t):
    return 231.37, f"OFFLINE TEST {t}"


def fake_hist(t, period="1y"):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end="2026-10-23", periods=300)
    return pd.Series(200 * np.exp(np.cumsum(rng.normal(0, 0.015, 300))), index=idx, name=t)


market.fetch_spot = fake_spot
market.fetch_history = fake_hist


class Upload(io.BytesIO):
    def __init__(self, path):
        super().__init__(open(path, "rb").read())
        self.name = os.path.basename(path)


def run(page, press=(), files=None, radio=None, keep_state=True, label=""):
    st.press = set(press)
    st.files = files or {}
    st.radio_choice = radio or {}
    st.calls = []
    if not keep_state:
        st.session_state.clear()
    for _ in range(3):  # re-run after st.rerun()
        try:
            runpy.run_path(os.path.join(ROOT, page), run_name="__main__")
            break
        except fake_streamlit.Stop:
            st.press = set()
            continue
    n_df = sum(1 for c in st.calls if c[0] == "df")
    n_ch = sum(1 for c in st.calls if c[0] == "chart")
    dls = [c[1] for c in st.calls if c[0] == "download"]
    errs = [c for c in st.calls if c[0] == "out" and "failed" in "".join(c[1:]).lower()]
    print(f"OK  {page:22s} {label:38s} tables={n_df:2d} charts={n_ch:2d} downloads={dls}")
    return st.calls


FAIL = []


def case(*a, **k):
    try:
        return run(*a, **k)
    except Exception:
        FAIL.append((a, k))
        print(f"FAIL {a[0]} {k.get('label', '')}")
        traceback.print_exc()


case("app.py", label="navigation")
case("views/home.py", label="home")

# ---- portfolio
case("views/portfolio.py", keep_state=False, label="default")
case("views/portfolio.py", press={"🌐 Fetch live spots"}, label="fetch spots")
case("views/portfolio.py", press={"📊 Estimate vols from history"}, label="estimate vols")
case("views/portfolio.py", press={"pf_buildpdf"}, label="build pdf")

# ---- arbitrage
case("views/arbitrage.py", keep_state=False, label="call overpriced (parity)")
case("views/arbitrage.py", press={"arb_fetch", "arb_sim_call", "arb_sim_put", "arb_buildpdf"}, label="fetch+MC+pdf")
st.session_state["arb_C"] = 8.0
case("views/arbitrage.py", press={"arb_buildpdf"}, label="call underpriced")
st.session_state["arb_C"] = 1.0
st.session_state["arb_P"] = 99.0
case("views/arbitrage.py", press={"arb_buildpdf"}, label="bounds violated")
st.session_state["arb_C"] = 10.450584
st.session_state["arb_P"] = 5.573526
case("views/arbitrage.py", label="fair prices")

# ---- hedging
case("views/hedging.py", keep_state=False, label="forecast GBM")
case("views/hedging.py", press={"hed_dl", "hedf_buildpdf", "hed_runmc"}, label="historical dl + mc + pdf")
case("views/hedging.py", press={"hedh_buildpdf"}, label="historical pdf")
case("views/hedging.py", radio={"hed_method": "Constant daily move"}, label="constant move")
case("views/hedging.py", radio={"hed_method": "Manual entry"}, label="manual entry")
case("views/hedging.py", radio={"hed_src": "Upload CSV/Excel"},
     files={"hed_up": Upload("data/sample_historical_prices.csv")}, press={"hedh_buildpdf"}, label="upload csv")
st.session_state["hed_type"] = "Put"
st.session_state["hed_pos"] = "Long"
st.session_state["hed_type2"] = "Put"
case("views/hedging.py", label="long put / put hedge")

# ---- sabr
case("views/sabr.py", keep_state=False, label="default")
case("views/sabr.py", files={"sabr_upload": Upload("data/sample_sabr_option_quotes.csv")}, press={"sabr_apply"},
     label="upload quotes csv + apply")
case("views/sabr.py", files={"sabr_upload": Upload("data/sample_sabr_option_quotes.xlsx")}, label="upload quotes xlsx")
case("views/sabr.py", files={"sabr_upload": Upload("data/sample_sabr_prices_only.csv")}, press={"sabr_buildpdf"},
     label="prices only + pdf")
case("views/sabr.py", files={"sabr_upload": Upload("data/sample_historical_prices.csv")}, press={"sabr_apply"},
     label="upload history + apply")
st.session_state["sabr_nsim"] = 20000
case("views/sabr.py", press={"sabr_run", "sabr_cmp", "sabr_mcsmile", "sabr_buildpdf"}, label="MC both schemes + pdf")
st.session_state["sabr_beta"] = 0.5
st.session_state["sabr_alpha"] = 0.27 * 230 ** 0.5
case("views/sabr.py", press={"sabr_run", "sabr_cmp", "sabr_usemkt"}, label="beta 0.5 MC")
case("views/sabr.py", radio={"sabr_calsrc": "Online price history (Yahoo)"}, press={"sabr_hdl"}, label="online history")
for k, v in {"sabr_beta": 1.0, "sabr_alpha": 0.27}.items():
    st.session_state[k] = v
st.session_state["sabr_mpx"] = 1.0
case("views/sabr.py", label="underpriced option")

# help pages render
from core import help_content  # noqa: E402

for k in ("portfolio", "arbitrage", "hedging", "sabr"):
    for sec in help_content.get_help(k):
        for kind, p in sec["blocks"]:
            if kind == "tex":
                help_content.latex_of(p)
print("help pages OK")

# dump the PDFs built during the run for visual inspection
out = os.environ.get("SMOKE_PDF_DIR")
if out:
    for k in ("pf_pdf", "arb_pdf", "hedf_pdf", "hedh_pdf", "sabr_pdf"):
        if st.session_state.get(k):
            open(os.path.join(out, k + ".pdf"), "wb").write(st.session_state[k])

print("FAILURES:", len(FAIL))
sys.exit(1 if FAIL else 0)
