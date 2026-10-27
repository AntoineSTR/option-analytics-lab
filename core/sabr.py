"""SABR model: Hagan et al. (2002) implied volatility, calibration and pricing (Module 4).

Dynamics (under the T-forward measure):
    dF_t     = alpha_t * F_t^beta * dW1_t
    dalpha_t = nu * alpha_t * dW2_t,          d<W1, W2>_t = rho dt

For a non-dividend-paying stock F = S e^{rT}. With beta = 1 (equities) the
model is lognormal in F with a stochastic volatility alpha_t.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import polygamma

from . import bs


# ---------------------------------------------------------------------------
# Hagan's formula
# ---------------------------------------------------------------------------
def hagan_vol(F, K, T, alpha, beta, rho, nu):
    """Black (lognormal) implied volatility from Hagan's asymptotic expansion."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    F, K = np.broadcast_arrays(F, K)
    rho = float(np.clip(rho, -0.9999, 0.9999))
    alpha = max(float(alpha), 1e-12)
    ob = 1.0 - beta
    logFK = np.log(F / K)
    FK = F * K
    FKb = FK ** (ob / 2.0)
    z = (nu / alpha) * FKb * logFK
    with np.errstate(divide="ignore", invalid="ignore"):
        xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
        zx = np.where(np.abs(z) < 1e-7, 1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho**2) * z**2 / 12.0, z / xz)
    denom = FKb * (1.0 + ob**2 / 24.0 * logFK**2 + ob**4 / 1920.0 * logFK**4)
    corr = 1.0 + (ob**2 / 24.0 * alpha**2 / FK**ob
                  + 0.25 * rho * beta * nu * alpha / FKb
                  + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * T
    vol = alpha / denom * zx * corr
    vol = np.asarray(vol, dtype=float)
    return float(vol) if vol.ndim == 0 else vol


def atm_vol(F, T, alpha, beta, rho, nu):
    return hagan_vol(F, F, T, alpha, beta, rho, nu)


def alpha_from_atm(sigma_atm, F, T, beta, rho, nu):
    """Solve the cubic in alpha so that the model reproduces a given ATM vol."""
    ob = 1.0 - beta
    a3 = ob**2 * T / (24.0 * F ** (2.0 * ob))
    a2 = rho * beta * nu * T / (4.0 * F**ob)
    a1 = 1.0 + (2.0 - 3.0 * rho**2) * nu**2 * T / 24.0
    a0 = -sigma_atm * F**ob
    roots = np.roots([a3, a2, a1, a0]) if abs(a3) > 1e-14 else np.roots([a2, a1, a0]) if abs(a2) > 1e-14 else np.array([-a0 / a1])
    real = roots[np.isreal(roots)].real
    real = real[real > 0]
    if real.size == 0:
        return sigma_atm * F**ob
    return float(real.min())


def sabr_price(S, K, T, r, alpha, beta, rho, nu, option="call"):
    F = S * np.exp(r * T)
    vol = hagan_vol(F, K, T, alpha, beta, rho, nu)
    return bs.black76(F, K, T, r, vol, option), vol


# ---------------------------------------------------------------------------
# Calibration to an observed smile
# ---------------------------------------------------------------------------
def calibrate_smile(strikes, vols, F, T, beta=1.0, weights=None) -> dict:
    K = np.asarray(strikes, dtype=float)
    v = np.asarray(vols, dtype=float)
    mask = np.isfinite(K) & np.isfinite(v) & (K > 0) & (v > 0)
    K, v = K[mask], v[mask]
    if K.size < 3:
        raise ValueError("At least 3 valid (strike, implied vol) points are needed to calibrate alpha, rho, nu.")
    w = np.ones_like(K) if weights is None else np.asarray(weights, dtype=float)[mask]
    order = np.argsort(K)
    atm_guess = float(np.interp(F, K[order], v[order]))
    a0 = atm_guess * F ** (1.0 - beta)

    def resid(p):
        a, rh, nu_ = p
        return w * (hagan_vol(F, K, T, a, beta, rh, nu_) - v)

    lb = [1e-6, -0.999, 1e-4]
    ub = [max(10.0 * a0, 5.0 * F ** (1.0 - beta)), 0.999, 10.0]
    best = None
    for rh0 in (-0.6, -0.2, 0.2):
        for nu0 in (0.3, 0.8, 1.5):
            try:
                sol = least_squares(resid, x0=[a0, rh0, nu0], bounds=(lb, ub), method="trf",
                                    xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
            except Exception:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
    if best is None:
        raise RuntimeError("Calibration failed.")
    a, rh, nu_ = best.x
    fitted = hagan_vol(F, K, T, a, beta, rh, nu_)
    err = fitted - v
    return {
        "alpha": float(a), "beta": float(beta), "rho": float(rh), "nu": float(nu_),
        "rmse": float(np.sqrt(np.mean(err**2))), "max_abs_error": float(np.max(np.abs(err))),
        "strikes": K, "market_vols": v, "fitted_vols": fitted, "success": bool(best.success),
        "message": str(best.message), "F": float(F), "T": float(T),
    }


# ---------------------------------------------------------------------------
# Calibration from a historical price series (statistical estimation)
# ---------------------------------------------------------------------------
def calibrate_historical(prices, beta=1.0, block=21, alpha_window=21, periods_per_year=252) -> dict:
    """Estimate alpha, rho, nu from daily closes.

    1. Split log-returns into non-overlapping blocks of `block` days.
    2. Realised vol of each block: sigma_j = sqrt(252/block * sum r^2).
    3. nu  = std of d ln sigma_j, annualised, minus the sampling noise of the
             realised-vol estimator: Var[ln sigma_hat] = psi'(n/2)/4 (trigamma).
    4. rho = corr(block return, d ln sigma_j), corrected for that same noise.
    5. alpha = current realised vol (last `alpha_window` days) * S^(1-beta).
    """
    p = pd.Series(prices).dropna().astype(float)
    p = p[p > 0]
    lr = np.log(p / p.shift(1)).dropna().values
    nb = len(lr) // block
    if nb < 8:
        raise ValueError(f"Not enough data: need at least {8 * block + 1} daily prices "
                         f"for block size {block} (got {len(p)}).")
    x = lr[-nb * block:].reshape(nb, block)
    rv = np.sqrt(np.mean(x**2, axis=1) * periods_per_year)
    rv = np.maximum(rv, 1e-8)
    ret_b = x.sum(axis=1)
    dl = np.diff(np.log(rv))
    # block return standardised by the vol known at the start of the block
    zb = ret_b[1:] / (rv[:-1] * np.sqrt(block / periods_per_year))
    var_meas = float(np.var(dl, ddof=1))
    noise = float(polygamma(1, block / 2.0)) / 4.0          # Var of ln(sigma_hat) per block
    var_true = var_meas - 2.0 * noise                         # differencing doubles the noise
    corrected = var_true > 0.05 * var_meas
    if not corrected:
        var_true = 0.05 * var_meas
    # Differences of block AVERAGES of a Brownian motion have variance (2/3) * block length,
    # so the per-unit-time variance of ln(alpha) is 1.5 * var_true / (block / ppy).
    nu = float(np.sqrt(1.5 * var_true * periods_per_year / block))
    rho_raw = float(np.corrcoef(zb, dl)[0, 1])
    # corr(block return, change of block-average log vol) = rho * sqrt(3/2) / 2 ; also undo the noise
    rho = float(np.clip(rho_raw * np.sqrt(var_meas / var_true) / (0.5 * np.sqrt(1.5)), -0.99, 0.99))
    sigma_now = float(np.sqrt(np.mean(lr[-alpha_window:] ** 2) * periods_per_year))
    S_last = float(p.iloc[-1])
    alpha = sigma_now * S_last ** (1.0 - beta)
    idx = p.index[1:][-nb * block:][block - 1::block] if isinstance(p.index, pd.DatetimeIndex) else np.arange(nb)
    details = pd.DataFrame({"Block end": idx[:nb], "Block return": ret_b, "Realised vol (ann.)": rv})
    return {
        "alpha": alpha, "beta": float(beta), "rho": rho, "nu": nu,
        "rho_uncorrected": rho_raw,
        "nu_uncorrected": float(np.sqrt(var_meas * periods_per_year / block)),
        "noise_correction_ok": bool(corrected),
        "sigma_now": sigma_now, "S_last": S_last, "n_blocks": nb, "details": details,
        "mean_realised_vol": float(np.mean(rv)),
    }


# ---------------------------------------------------------------------------
# Data loading (CSV / Excel)
# ---------------------------------------------------------------------------
_ALIASES = {
    "strike": ["strike", "k", "strike price"],
    "iv": ["implied vol", "implied volatility", "impliedvolatility", "iv", "vol", "volatility", "sigma", "implied_vol"],
    "price": ["market price", "price", "mid", "premium", "last", "lastprice", "option price", "market_price"],
    "type": ["type", "option type", "cp", "call/put", "option_type", "right"],
    "T": ["maturity (years)", "maturity", "t", "time to maturity", "ttm", "tau", "expiry (years)"],
    "spot": ["spot", "s", "underlying", "underlying price", "s0"],
    "forward": ["forward", "f", "forward price"],
    "rate": ["rate", "r", "risk-free rate", "risk free rate"],
    "date": ["date", "datetime", "time", "day"],
    "close": ["close", "adj close", "adj_close", "price", "close price", "last"],
}


def read_table(uploaded) -> pd.DataFrame:
    name = getattr(uploaded, "name", str(uploaded)).lower()
    data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else open(uploaded, "rb").read()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    txt = data.decode("utf-8-sig", errors="replace")
    sep = ";" if txt.count(";") > txt.count(",") else ","
    return pd.read_csv(io.StringIO(txt), sep=sep)


def _find(df, key):
    cols = {c.strip().lower(): c for c in df.columns}
    for a in _ALIASES[key]:
        if a in cols:
            return cols[a]
    return None


def parse_smile(df: pd.DataFrame, S=None, r=0.0, T=None) -> tuple[pd.DataFrame, dict]:
    """Normalise an option-quote table to columns Strike, Type, Price, IV, T, Spot, Forward."""
    info = {}
    ks = _find(df, "strike")
    if ks is None:
        raise ValueError("The file needs a 'Strike' column.")
    out = pd.DataFrame({"Strike": pd.to_numeric(df[ks], errors="coerce")})
    tcol, scol, fcol, rcol = _find(df, "T"), _find(df, "spot"), _find(df, "forward"), _find(df, "rate")
    out["T"] = pd.to_numeric(df[tcol], errors="coerce") if tcol else T
    if rcol:
        rv = pd.to_numeric(df[rcol], errors="coerce").dropna()
        if len(rv):
            r = float(rv.iloc[0])
            if r > 1:
                r /= 100.0
            info["rate_from_file"] = r
    if scol:
        out["Spot"] = pd.to_numeric(df[scol], errors="coerce")
    else:
        out["Spot"] = S
    if fcol:
        out["Forward"] = pd.to_numeric(df[fcol], errors="coerce")
    else:
        out["Forward"] = out["Spot"] * np.exp(r * out["T"].astype(float))
    if out["Spot"].isna().all() and fcol:
        out["Spot"] = out["Forward"] * np.exp(-r * out["T"].astype(float))
    ty = _find(df, "type")
    out["Type"] = df[ty].astype(str).str.strip().str.lower().str[0].map({"c": "Call", "p": "Put"}).fillna("Call") if ty else "Call"
    pc = _find(df, "price")
    ivc = _find(df, "iv")
    out["Price"] = pd.to_numeric(df[pc], errors="coerce") if pc else np.nan
    if ivc:
        iv = pd.to_numeric(df[ivc], errors="coerce")
        if iv.median() > 3:  # given in percent
            iv = iv / 100.0
            info["iv_percent_converted"] = True
        out["IV"] = iv
    else:
        out["IV"] = np.nan
    need = out["IV"].isna() & out["Price"].notna()
    if need.any():
        if out["Spot"].isna().any() or out["T"].isna().any():
            raise ValueError("Prices without implied vols need Spot and Maturity (column or input) to be inverted.")
        out.loc[need, "IV"] = [
            bs.implied_vol(p, s, k, t, r, ty_)
            for p, s, k, t, ty_ in out.loc[need, ["Price", "Spot", "Strike", "T", "Type"]].itertuples(index=False)
        ]
        info["iv_from_prices"] = int(need.sum())
    if out["Price"].isna().all() and out["Spot"].notna().all() and out["T"].notna().all():
        out["Price"] = [bs.price(s, k, t, r, v, ty_) for s, k, t, v, ty_ in
                        out[["Spot", "Strike", "T", "IV", "Type"]].itertuples(index=False)]
    out = out.dropna(subset=["Strike", "IV"]).sort_values(["T", "Strike"]).reset_index(drop=True)
    info["rate"] = r
    return out, info


def parse_history(df: pd.DataFrame) -> pd.Series:
    dc, cc = _find(df, "date"), _find(df, "close")
    if cc is None:
        num = df.select_dtypes("number").columns
        if len(num) == 0:
            raise ValueError("No price column found (expected 'Close').")
        cc = num[-1]
    s = pd.to_numeric(df[cc], errors="coerce")
    if dc:
        s.index = pd.to_datetime(df[dc], errors="coerce")
        s = s[s.index.notna()].sort_index()
    return s.dropna()


def is_history_table(df: pd.DataFrame) -> bool:
    return _find(df, "strike") is None and (_find(df, "close") is not None or _find(df, "date") is not None)


# ---------------------------------------------------------------------------
# Smile analytics
# ---------------------------------------------------------------------------
def smile_grid(alpha, beta, rho, nu, spots, strikes, r, T) -> pd.DataFrame:
    """Implied vol for every (S, K): F = S e^{rT}. Rows = spots, columns = strikes."""
    data = {}
    for S in spots:
        F = S * np.exp(r * T)
        data[S] = hagan_vol(F, np.asarray(strikes, float), T, alpha, beta, rho, nu)
    g = pd.DataFrame(data, index=np.round(strikes, 4)).T
    g.index.name = "Spot S"
    g.insert(0, "Forward F", [S * np.exp(r * T) for S in spots])
    return g


def pricing_table(S, r, T, alpha, beta, rho, nu, strikes, option="call", flat_vol=None, market=None) -> pd.DataFrame:
    F = S * np.exp(r * T)
    K = np.asarray(strikes, dtype=float)
    vol = hagan_vol(F, K, T, alpha, beta, rho, nu)
    if flat_vol is None:
        flat_vol = atm_vol(F, T, alpha, beta, rho, nu)
    sabr_px = bs.black76(F, K, T, r, vol, option)
    bs_px = bs.price(S, K, T, r, flat_vol, option)
    df = pd.DataFrame({
        "Strike": K, "K/F": K / F, "Spot": S, "Forward": F, "T": T,
        "SABR implied vol": vol, f"{option.capitalize()} price (SABR/Hagan)": sabr_px,
        "Flat BS vol": flat_vol, f"{option.capitalize()} price (BS flat vol)": bs_px,
        "SABR - BS": sabr_px - bs_px,
        "SABR delta": bs.black76_delta(F, K, T, r, vol, option),
    })
    if market is not None:
        m = np.asarray(market, dtype=float)
        df["Market price"] = m
        df["Market - SABR"] = m - sabr_px
        df["Signal"] = np.where(np.isnan(m), "",
                                np.where(m - sabr_px > 0, "Overpriced -> SELL", "Underpriced -> BUY"))
    return df


def density_check(S, r, T, alpha, beta, rho, nu, kmin=None, kmax=None, n=400) -> pd.DataFrame:
    """Risk-neutral density from Breeden-Litzenberger: q(K) = e^{rT} d2C/dK2.
    Negative values = butterfly (static) arbitrage in the Hagan smile."""
    F = S * np.exp(r * T)
    kmin = kmin or 0.3 * F
    kmax = kmax or 2.0 * F
    K = np.linspace(kmin, kmax, n)
    h = K[1] - K[0]
    C = bs.black76(F, K, T, r, hagan_vol(F, K, T, alpha, beta, rho, nu), "call")
    d2 = np.full_like(K, np.nan)
    d2[1:-1] = (C[2:] - 2 * C[1:-1] + C[:-2]) / h**2
    return pd.DataFrame({"Strike": K, "Call price": C, "Density": np.exp(r * T) * d2,
                         "Butterfly": np.r_[np.nan, C[2:] - 2 * C[1:-1] + C[:-2], np.nan]})


def make_sample_smile(S=230.0, r=0.045, T=0.25, alpha=0.27, beta=1.0, rho=-0.55, nu=1.1, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    F = S * np.exp(r * T)
    K = np.round(np.linspace(0.75 * S, 1.25 * S, 21) / 5) * 5
    K = np.unique(K)
    iv = hagan_vol(F, K, T, alpha, beta, rho, nu) + rng.normal(0, 0.002, K.size)
    typ = np.where(K < S, "Put", "Call")
    px = [bs.price(S, k, T, r, v, t) for k, v, t in zip(K, iv, typ)]
    return pd.DataFrame({"Strike": K, "Type": typ, "Market Price": np.round(px, 4),
                         "Implied Vol": np.round(iv, 5), "Maturity (years)": T, "Spot": S, "Rate": r})
