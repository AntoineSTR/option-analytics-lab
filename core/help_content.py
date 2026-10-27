"""Help content for every module.

Every formula is a SymPy expression converted to LaTeX with ``sympy.latex``.
Expressions are built with ``evaluate=False`` so SymPy keeps the textbook
ordering of terms. The SABR reductions (beta = 1 and at-the-money) are
derived symbolically from the general Hagan expression and the equality is
checked with ``sympy.simplify`` (see ``verify_sabr_reductions``).

get_help(key) -> list of sections; section = {"title": str, "blocks": [(kind, payload), ...]}
kind = "md" (markdown) or "tex" (SymPy expression).
"""
from __future__ import annotations

from functools import lru_cache

import sympy as sp

# ---------------------------------------------------------------------------
# Builders that keep the written order of terms
# ---------------------------------------------------------------------------
def Mu(*a):
    return sp.Mul(*a, evaluate=False)


def Ad(*a):
    return sp.Add(*a, evaluate=False)


def neg(a):
    return Mu(-1, a)


def frac(a, b):
    return Mu(a, sp.Pow(b, -1, evaluate=False))


def Pw(a, b):
    return sp.Pow(a, b, evaluate=False)


def eq(lhs, rhs):
    return sp.Eq(lhs, rhs, evaluate=False)


def L(expr) -> str:
    return sp.latex(expr, order="none", ln_notation=True)


def sym(name, **kw):
    return sp.Symbol(name, **kw)


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
S, K, T, r, t = sp.symbols("S K T r t", positive=True)
sigma, mu = sp.symbols("sigma mu", positive=True)
C, P = sp.symbols("C P")
d1, d2 = sp.symbols("d_1 d_2")
N = sp.Function("N")
phi = sp.Function("varphi")
S_T = sym("S_T", positive=True)
q = sym("q")
Delta, Gamma, B = sp.symbols("Delta Gamma B")
Delta1, Delta2, Gamma1, Gamma2 = sp.symbols("Delta_1 Delta_2 Gamma_1 Gamma_2")
h, w, Pi = sp.symbols("h w Pi")
dt = sym(r"\Delta t", positive=True)
Z = sym("Z")
F, alpha, beta, rho, nu = sp.symbols("F alpha beta rho nu", positive=True)
z, x = sp.symbols("z x")
dF, dalpha, dW1, dW2, dtt = sp.symbols(r"dF d\alpha dW_1 dW_2 dt")
W1, W2 = sp.symbols("W_1 W_2")
i = sym("i", integer=True)
M = sym("M", positive=True)
m = sym("m", positive=True)

DISC = sp.exp(Mu(-1, r, T))                      # e^{-rT}
PVK = Mu(K, DISC)                                # K e^{-rT}
D1 = frac(Ad(sp.log(S / K), Mu(Ad(r, frac(sigma**2, 2)), T)), Mu(sigma, sp.sqrt(T)))
D2 = Ad(d1, neg(Mu(sigma, sp.sqrt(T))))
CALL = Ad(Mu(S, N(d1)), neg(Mu(PVK, N(d2))))
PUT = Ad(Mu(PVK, N(neg(d2))), neg(Mu(S, N(neg(d1)))))

BS_BLOCK = [
    ("md", "**Black-Scholes-Merton price** (European option, no dividends):"),
    ("tex", eq(C, CALL)),
    ("tex", eq(P, PUT)),
    ("tex", eq(d1, D1)),
    ("tex", eq(d2, D2)),
    ("md", "$N(\\cdot)$ is the standard normal cumulative distribution and $\\varphi(\\cdot)$ its density."),
]

GREEKS_BLOCK = [
    ("md", "**Greeks** (sensitivities of the option price $V$):"),
    ("tex", eq(sym(r"\Delta_{call}"), sp.Eq(sp.Derivative(sym("V"), S), N(d1), evaluate=False))),
    ("tex", eq(sym(r"\Delta_{put}"), Ad(N(d1), -1))),
    ("tex", eq(Gamma, sp.Eq(sp.Derivative(sym("V"), (S, 2)), frac(phi(d1), Mu(S, sigma, sp.sqrt(T))), evaluate=False))),
    ("tex", eq(sym(r"\mathcal{V}"), Mu(S, phi(d1), sp.sqrt(T)))),
    ("tex", eq(sym(r"\Theta_{call}"),
               Ad(neg(frac(Mu(S, phi(d1), sigma), Mu(2, sp.sqrt(T)))), neg(Mu(r, PVK, N(d2)))))),
]


# ---------------------------------------------------------------------------
# SABR symbolic objects
# ---------------------------------------------------------------------------
def _hagan_general():
    """Hagan's formula as one (evaluated) SymPy expression of F, K, T, alpha, beta, rho, nu."""
    ob = 1 - beta
    lfk = sp.log(F / K)
    FKb = (F * K) ** (ob / 2)
    zz = nu / alpha * FKb * lfk
    xz = sp.log((sp.sqrt(1 - 2 * rho * zz + zz**2) + zz - rho) / (1 - rho))
    corr = 1 + (ob**2 / 24 * alpha**2 / (F * K) ** ob + rho * beta * nu * alpha / (4 * FKb)
                + (2 - 3 * rho**2) / 24 * nu**2) * T
    pre = alpha / (FKb * (1 + ob**2 / 24 * lfk**2 + ob**4 / 1920 * lfk**4))
    return pre, zz, xz, corr


def _hagan_display():
    ob = Ad(1, neg(beta))
    lfk = sp.log(F / K)
    FK = Mu(F, K)
    FKb = Pw(FK, frac(ob, 2))
    D = Ad(1, Mu(frac(Pw(ob, 2), 24), Pw(lfk, 2)), Mu(frac(Pw(ob, 4), 1920), Pw(lfk, 4)))
    A = Ad(frac(Mu(Pw(ob, 2), alpha**2), Mu(24, Pw(FK, ob))),
           frac(Mu(rho, beta, nu, alpha), Mu(4, FKb)),
           frac(Mu(Ad(2, neg(Mu(3, rho**2))), nu**2), 24))
    sigB = Mu(frac(alpha, Mu(FKb, D)), frac(z, x), Ad(1, Mu(A, T)))
    zdef = Mu(frac(nu, alpha), FKb, lfk)
    xdef = sp.log(frac(Ad(sp.sqrt(Ad(1, neg(Mu(2, rho, z)), z**2)), z, neg(rho)), Ad(1, neg(rho))))
    return sigB, zdef, xdef


def _hagan_beta1_display():
    A = Ad(frac(Mu(rho, nu, alpha), 4), frac(Mu(Ad(2, neg(Mu(3, rho**2))), nu**2), 24))
    return Mu(alpha, frac(z, x), Ad(1, Mu(A, T))), Mu(frac(nu, alpha), sp.log(F / K))


def _hagan_atm_display():
    ob = Ad(1, neg(beta))
    A = Ad(frac(Mu(Pw(ob, 2), alpha**2), Mu(24, Pw(F, Mu(2, ob)))),
           frac(Mu(rho, beta, nu, alpha), Mu(4, Pw(F, ob))),
           frac(Mu(Ad(2, neg(Mu(3, rho**2))), nu**2), 24))
    return Mu(frac(alpha, Pw(F, ob)), Ad(1, Mu(A, T)))


@lru_cache(maxsize=1)
def verify_sabr_reductions() -> dict:
    """Check with SymPy that the displayed beta=1 and ATM formulas follow from Hagan's general formula."""
    pre, zz, xz, corr = _hagan_general()
    general = pre * (z / x) * corr                       # z, x kept symbolic as in the paper
    beta1 = sp.simplify(general.subs(beta, 1) - _hagan_beta1_display()[0].doit())
    z_b1 = sp.simplify(zz.subs(beta, 1) - _hagan_beta1_display()[1].doit())
    atm_general = (pre * corr).subs(K, F)               # z/x(z) -> 1 when K -> F
    atm = sp.simplify(atm_general - _hagan_atm_display().doit())
    limit_zx = sp.limit(z / sp.log((sp.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho)), z, 0)
    return {"beta1": beta1 == 0 and z_b1 == 0, "atm": atm == 0, "limit_z_over_x": limit_zx}


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
def _portfolio():
    V, Vi, qi, si, n = sp.symbols("V V_i q_i s_i n")
    return [
        {"title": "What this module does", "blocks": [("md", (
            "It values a **portfolio of European options on stocks** (calls and puts, long or short). For each "
            "position it computes the **intrinsic value**, the **Black-Scholes value**, the **time value** and the "
            "**market value** of the position, plus the portfolio Greeks (risk exposures). The spot price of each "
            "underlying can be **fetched online** (Yahoo Finance) or **typed manually**."))]},
        {"title": "How to use it", "blocks": [("md", (
            "1. Edit the positions table: ticker, spot, option type (Call/Put), position (Long/Short), quantity "
            "(number of shares covered = number of options on one share), strike, maturity in years, volatility in %.\n"
            "2. Click **Fetch live spots** to replace the spots with online prices (or keep your manual values). "
            "**Estimate vols from history** fills the volatility column with 1-year historical volatility.\n"
            "3. Optional: *Market price* = observed option quote (used instead of the model value); "
            "*Premium paid/received* = your entry price per option, used for the unrealised P&L.\n"
            "4. Read the results and charts, then download the PDF / Excel report."))]},
        {"title": "Formulas", "blocks": [
            ("md", "**Payoff at maturity** and **intrinsic value** (value if exercised now):"),
            ("tex", eq(sym(r"\mathrm{Payoff}_{call}"), sp.Max(Ad(S_T, neg(K)), 0))),
            ("tex", eq(sym(r"\mathrm{Payoff}_{put}"), sp.Max(Ad(K, neg(S_T)), 0))),
            ("tex", eq(sym(r"\mathrm{Intrinsic}_{call}"), sp.Max(Ad(S, neg(K)), 0))),
            ("tex", eq(sym(r"\mathrm{Intrinsic}_{put}"), sp.Max(Ad(K, neg(S)), 0))),
            *BS_BLOCK,
            ("md", "**Time value** and **market value of the portfolio** ($s_i = +1$ long, $-1$ short, "
                   "$q_i$ quantity, $V_i$ value of one option):"),
            ("tex", eq(sym(r"\mathrm{Time\ value}"), Ad(V, neg(sym(r"\mathrm{Intrinsic}"))))),
            ("tex", eq(sym(r"\mathrm{MV}"), sp.Sum(Mu(si, qi, Vi), (i, 1, n)))),
            *GREEKS_BLOCK,
        ]},
        {"title": "Assumptions", "blocks": [("md", (
            "- European exercise (only at maturity), no dividends.\n"
            "- Lognormal stock price, constant volatility and interest rate (Black-Scholes world).\n"
            "- Continuous compounding; maturity in years (3 months = 0.25).\n"
            "- One option = right on **one** share. A listed contract usually covers 100 shares: quantity = "
            "100 x number of contracts.\n"
            "- Market value uses the market quote if provided, otherwise the Black-Scholes value (mark-to-model)."))]},
        {"title": "How to read the results", "blocks": [("md", (
            "- **ITM / ATM / OTM**: in, at, out of the money. An OTM option has zero intrinsic value: all its value is "
            "time value, which decays to 0 at maturity.\n"
            "- **Delta (shares)**: the portfolio behaves like this many shares for small moves.\n"
            "- **Gamma**: how fast delta changes. **Vega**: P&L for +1 volatility point. **Theta**: value lost per day."))]},
    ]


def _arbitrage():
    Csyn, Psyn, prof = sp.symbols(r"C_{syn} P_{syn} \pi_0")
    return [
        {"title": "What this module does", "blocks": [("md", (
            "It checks whether option prices allow an **arbitrage**: a riskless profit with no net investment. It "
            "builds **synthetic options** from the stock and zero-coupon bonds (debt) and compares them with market "
            "prices. When an option is over- or under-priced it prints the exact **step-by-step strategy** (what to "
            "buy, sell, borrow or lend) with the cash flows today and at maturity, and verifies it on a payoff chart."))]},
        {"title": "How to use it", "blocks": [("md", (
            "1. Enter S (or fetch it online), K, T, r.\n2. Enter the market prices of the call and/or the put.\n"
            "3. Optional: a threshold (e.g. transaction costs) below which gaps are ignored, the number of options, and "
            "a volatility for the model-based check.\n4. A true arbitrage has a **positive cash flow today** and a "
            "**non-negative payoff in every scenario** at maturity: the app shows both."))]},
        {"title": "1. Put-call parity (model-free)", "blocks": [
            ("tex", eq(Ad(C, neg(P)), Ad(S, neg(PVK)))),
            ("md", "A call can therefore be **replicated** with a put, the stock and debt (and vice versa):"),
            ("tex", eq(Csyn, Ad(P, S, neg(PVK)))),
            ("tex", eq(Psyn, Ad(C, PVK, neg(S)))),
            ("md", "If $C > C_{syn}$: **sell the call and buy the synthetic call** (buy the put, buy the stock, "
                   "borrow $Ke^{-rT}$). If $C < C_{syn}$: the reverse. Profit locked today per option:"),
            ("tex", eq(prof, sp.Abs(Ad(C, neg(Csyn))))),
            ("md", "At maturity every leg offsets exactly, whatever $S_T$."),
        ]},
        {"title": "2. No-arbitrage bounds (model-free)", "blocks": [
            ("tex", sp.Le(sp.Max(0, Ad(S, neg(PVK))), C, evaluate=False)),
            ("tex", sp.Le(C, S, evaluate=False)),
            ("tex", sp.Le(sp.Max(0, Ad(PVK, neg(S))), P, evaluate=False)),
            ("tex", sp.Le(P, PVK, evaluate=False)),
            ("md", "A price outside these bounds gives a strategy with positive cash today and a payoff >= 0."),
        ]},
        {"title": "3. Black-Scholes replication (model-dependent)", "blocks": [
            ("md", "A call is equivalent to $\\Delta$ shares financed by borrowing $B$ (a put: short shares + lending):"),
            ("tex", eq(C, Ad(Mu(Delta, S), neg(B)))),
            ("tex", eq(Delta, N(d1))),
            ("tex", eq(B, Mu(PVK, N(d2)))),
            ("md", "If the market price differs from this value, sell the expensive side, buy the cheap one and "
                   "**re-balance the hedge** over time (module 3). The profit is only guaranteed if the realised "
                   "volatility equals the model volatility: it is **not** a pure arbitrage."),
            *BS_BLOCK,
        ]},
        {"title": "Assumptions", "blocks": [("md", (
            "- European options, non-dividend-paying stock, same strike and maturity for the call and the put.\n"
            "- Borrowing and lending at the risk-free rate r (zero-coupon bond paying K at T).\n"
            "- Short-selling allowed, no transaction costs other than the threshold you choose.\n"
            "- Prices are executable (in practice: sell at the bid, buy at the ask)."))]},
    ]


def _hedging():
    Bf, hf, Sf = sp.Function("B"), sp.Function("h"), sp.Function("S")
    tdt = Ad(t, dt)
    V1, V2 = sp.symbols("V_1 V_2")
    return [
        {"title": "What this module does", "blocks": [("md", (
            "A trader who sells (or buys) options is exposed to the stock price. **Delta hedging** neutralises small "
            "moves by holding shares; **delta-gamma hedging** also neutralises the curvature (gamma) with a second "
            "option. Delta and gamma change every day, so the hedge must be **re-balanced**: this module computes the "
            "daily adjustments, the cash account and the P&L, on **historical** prices or on **simulated** future "
            "prices (Spot Price Simulator, up to 5 days)."))]},
        {"title": "How to use it", "blocks": [("md", (
            "1. Enter K, S (online or manual), r, sigma, T, the option type, your position and the number of options.\n"
            "2. For delta-gamma hedging choose the hedge option (strike K2, maturity T2).\n"
            "3. **Historical tab**: downloads the last N daily closes (or upload a CSV) and replays the hedge day by day.\n"
            "4. **Forecast tab**: the simulator generates 1-5 daily spots (random GBM, constant daily move or your own "
            "values, editable in the table) and projects the P&L. The Monte Carlo box shows the P&L distribution over "
            "thousands of simulated paths.\n5. Download the daily adjustment report (PDF / Excel)."))]},
        {"title": "Formulas", "blocks": [
            ("md", "Option position $q = \\pm n$ (positive long, negative short). **Delta hedge** (shares held):"),
            ("tex", eq(h, neg(Mu(q, Delta)))),
            ("md", "**Delta-gamma hedge**: neutralise gamma with $w$ units of option 2, then delta with shares:"),
            ("tex", eq(w, neg(frac(Mu(q, Gamma1), Gamma2)))),
            ("tex", eq(h, neg(Ad(Mu(q, Delta1), Mu(w, Delta2))))),
            ("md", "**Self-financing cash account** (re-balancing trades are paid from the cash account):"),
            ("tex", eq(Bf(tdt), Ad(Mu(Bf(t), sp.exp(Mu(r, dt))), neg(Mu(Ad(hf(tdt), neg(hf(t))), Sf(tdt)))))),
            ("tex", eq(Pi, Ad(Mu(q, V1), Mu(w, V2), Mu(h, S), B))),
            ("md", "$\\Pi_0 = 0$, so $\\Pi_t$ is directly the cumulative P&L of the hedged book. Why a delta-hedged "
                   "book still makes or loses money (gamma-theta trade-off, realised vs implied variance):"),
            ("tex", eq(sym(r"\mathrm{P\&L}_{\Delta t}"),
                       Mu(sp.Rational(1, 2), q, Gamma, S**2, Ad(Pw(frac(sym(r"\Delta S"), S), 2), neg(Mu(sigma**2, dt)))))),
            ("md", "**Spot Price Simulator** (geometric Brownian motion, exact discretisation, $Z \\sim N(0,1)$):"),
            ("tex", eq(Sf(tdt), Mu(Sf(t), sp.exp(Ad(Mu(Ad(mu, neg(frac(sigma**2, 2))), dt), Mu(sigma, sp.sqrt(dt), Z)))))),
            *GREEKS_BLOCK,
        ]},
        {"title": "Assumptions", "blocks": [("md", (
            "- Black-Scholes prices and Greeks with the volatility you enter; European options, no dividends.\n"
            "- Re-balancing once per day at the closing price; 252 trading days per year (editable).\n"
            "- Shares and options are perfectly divisible; optional proportional transaction costs.\n"
            "- In the simulator the drift mu and the simulated volatility can differ from the pricing volatility: the "
            "gap between realised and implied volatility is what drives the hedged P&L."))]},
    ]


def _sabr():
    sB = sym(r"\sigma_B")
    sigB, zdef, xdef = _hagan_display()
    sigB1, z1def = _hagan_beta1_display()
    atm = _hagan_atm_display()
    chk = verify_sabr_reductions()
    Ff, af, g = sp.Function("F"), sp.Function(r"\alpha"), sp.Function("g")
    tdt = Ad(t, dt)
    Ki = sym("K_i")
    obj = sp.Sum(Mu(sym("w_i"), Pw(Ad(sp.Function(r"\sigma_{SABR}")(Ki), neg(sym(r"\sigma_i^{mkt}"))), 2)),
                 (i, 1, sym("n")))
    sj = sym(r"\hat{\sigma}_j")
    psi1 = sp.Function(r"\psi_1")
    Cf = sp.Function("C")(K)
    reduction_note = (
        "SymPy check: substituting beta = 1 (and K = F) into the general formula and simplifying gives exactly the "
        f"two expressions below - verified: beta=1 **{chk['beta1']}**, ATM **{chk['atm']}**, "
        f"and the limit of z/x(z) as z -> 0 is **{chk['limit_z_over_x']}**.")
    return [
        {"title": "What this module does", "blocks": [("md", (
            "Black-Scholes assumes one constant volatility, but real option prices show a **volatility smile / skew**: "
            "implied volatility depends on the strike. **SABR** (Stochastic Alpha Beta Rho) makes the volatility "
            "random and correlated with the price, which reproduces the smile. This module computes SABR implied vols "
            "with **Hagan's formula**, **calibrates** alpha, rho, nu from CSV/Excel data (beta = 1 for equities), plots "
            "the smile for several spot / forward levels, prices options, flags **mispriced** options with an arbitrage "
            "strategy, and **simulates the SABR SDEs by Monte Carlo**."))]},
        {"title": "How to use it", "blocks": [("md", (
            "1. **Calculator**: enter S (or fetch), r, T, K and the parameters: Hagan implied vol, price, BS comparison.\n"
            "2. **Calibration**: upload a CSV / Excel file (sample files can be downloaded):\n"
            "   - *Option quotes*: Strike and Implied Vol (or Market Price + Type), optional Maturity, Spot, Rate -> "
            "least-squares fit of the smile.\n"
            "   - *Historical prices*: Date and Close -> statistical estimation of alpha, rho, nu from the dynamics "
            "of realised volatility.\n"
            "3. **Smile & S/F grid**: smiles for several spot levels (F = S e^(rT)) and the full table.\n"
            "4. **Pricing & arbitrage**: SABR vs Black-Scholes table, strategy for a mispriced option, smile arbitrage check.\n"
            "5. **Monte Carlo**: scheme, number of simulations and steps -> expected final stock price and present "
            "value of the average payoff."))]},
        {"title": "The SABR model", "blocks": [
            ("tex", eq(dF, Mu(alpha, Pw(F, beta), dW1))),
            ("tex", eq(dalpha, Mu(nu, alpha, dW2))),
            ("tex", eq(Mu(dW1, dW2), Mu(rho, dtt))),
            ("tex", eq(F, Mu(S, sp.exp(Mu(r, T))))),
            ("md", "- **alpha**: volatility level. **beta**: elasticity (1 = lognormal, used for equities). "
                   "**rho**: price/volatility correlation (negative = skew, low strikes more expensive). "
                   "**nu**: volatility of volatility (curvature of the smile)."),
        ]},
        {"title": "Hagan's implied volatility formula (2002)", "blocks": [
            ("tex", eq(sB, sigB)),
            ("tex", eq(z, zdef)),
            ("tex", eq(x, xdef)),
            ("md", reduction_note),
            ("md", "**With beta = 1** (equities):"),
            ("tex", eq(sB, sigB1)),
            ("tex", eq(z, z1def)),
            ("md", "**At the money** ($K = F$):"),
            ("tex", eq(sym(r"\sigma_{ATM}"), atm)),
            ("md", "The option is then priced with Black's formula on the forward (identical to Black-Scholes with "
                   "the SABR volatility):"),
            ("tex", eq(C, Mu(DISC, Ad(Mu(F, N(d1)), neg(Mu(K, N(d2))))))),
            ("tex", eq(d1, frac(Ad(sp.log(F / K), frac(Mu(sB**2, T), 2)), Mu(sB, sp.sqrt(T))))),
        ]},
        {"title": "Calibration", "blocks": [
            ("md", "**From option quotes**: beta fixed (1 for equities), non-linear least squares (trust-region "
                   "reflective, several starting points):"),
            ("tex", Mu(sym(r"\min_{\alpha,\ \rho,\ \nu}"), obj)),
            ("md", "**From historical prices**: split daily log-returns $r_k$ into blocks of $m$ days (default 21), "
                   "compute each block's realised volatility, then:"),
            ("tex", eq(sj, sp.sqrt(Mu(frac(sym("252"), m), sp.Sum(sym("r_k") ** 2, (sym("k"), 1, m)))))),
            ("tex", eq(sym(r"\hat{\nu}"), Pw(Mu(sp.Rational(3, 2), frac(sym("252"), m),
                                                  Ad(sym(r"\mathrm{Var}[\Delta \ln \hat{\sigma}_j]"),
                                                     neg(frac(psi1(frac(m, 2)), 2)))), sp.Rational(1, 2)))),
            ("tex", eq(sym(r"\hat{\rho}"), Mu(sp.Pow(sp.Rational(8, 3), sp.Rational(1, 2), evaluate=False), sym(r"\mathrm{Corr}(R_j,\ \Delta \ln \hat{\sigma}_j)"),
                                              sp.sqrt(frac(sym(r"\mathrm{Var}_{meas}"), sym(r"\mathrm{Var}_{true}")))))),
            ("tex", eq(sym(r"\hat{\alpha}"), Mu(sym(r"\sigma_{realised}^{21d}"), Pw(S, Ad(1, neg(beta)))))),
            ("md", "$\\psi_1$ (trigamma) removes the sampling noise of realised volatility; the factors 3/2 and "
                   "$\\sqrt{8/3}$ correct for the averaging of volatility inside each block ($R_j$ = block return "
                   "standardised by the previous block's volatility). Use 3+ years of data. This is a statistical "
                   "(real-world) estimate; calibration to option quotes is preferred when available."),
        ]},
        {"title": "Arbitrage checks", "blocks": [
            ("md", "**Mispriced option**: if the market price is above the SABR price, sell it and buy the replicating "
                   "portfolio (delta hedge at the SABR volatility); if below, buy it and sell the replicating portfolio. "
                   "**Static (butterfly) arbitrage** of the smile is checked with the Breeden-Litzenberger risk-neutral "
                   "density, which must be non-negative:"),
            ("tex", eq(sp.Function("q")(K), Mu(sp.exp(Mu(r, T)), sp.Derivative(Cf, (K, 2))))),
            ("md", "Hagan's formula is an approximation and can give a negative density for very low strikes, long "
                   "maturities or large nu: the app reports where."),
        ]},
        {"title": "Monte Carlo simulation of the SABR SDEs", "blocks": [
            ("md", "Correlated normals: $W_1 = Z_1$, $W_2 = \\rho Z_1 + \\sqrt{1-\\rho^2}\\,Z_2$. "
                   "**Standard Euler-Maruyama**:"),
            ("tex", eq(Ff(tdt), Ad(Ff(t), Mu(af(t), Pw(Ff(t), beta), sp.sqrt(dt), W1)))),
            ("tex", eq(af(tdt), Ad(af(t), Mu(nu, af(t), sp.sqrt(dt), W2)))),
            ("md", "**Log transform / full truncation (production standard)**: alpha is simulated exactly in log space "
                   "(always positive); for beta = 1 the forward too (log-Euler, F > 0); for beta < 1 only the positive "
                   "part of F enters the diffusion (full truncation) and F_T is floored at 0:"),
            ("tex", eq(af(tdt), Mu(af(t), sp.exp(Ad(Mu(nu, sp.sqrt(dt), W2), neg(frac(Mu(nu**2, dt), 2))))))),
            ("tex", eq(Ff(tdt), Mu(Ff(t), sp.exp(Ad(Mu(af(t), sp.sqrt(dt), W1), neg(frac(Mu(Pw(af(t), 2), dt), 2))))))),
            ("tex", eq(Ff(tdt), Ad(Ff(t), Mu(af(t), Pw(sp.Max(Ff(t), 0), beta), sp.sqrt(dt), W1)))),
            ("md", "At maturity $F_T = S_T$. **Final price of the underlying** and **option value** "
                   "($M$ simulations, $g$ = payoff):"),
            ("tex", eq(sym(r"\hat{E}[S_T]"), frac(sp.Sum(sym("S_T^{(i)}"), (i, 1, M)), M))),
            ("tex", eq(sym("V_0"), Mu(DISC, frac(sp.Sum(g(sym("S_T^{(i)}")), (i, 1, M)), M)))),
            ("tex", eq(sym(r"\mathrm{SE}"), frac(sym("s"), sp.sqrt(M)))),
            ("md", "95% confidence interval: price +/- 1.96 SE. **Antithetic variates** (Z and -Z) reduce the "
                   "variance. The error falls like $1/\\sqrt{M}$: 4x more paths halves it."),
        ]},
        {"title": "Assumptions", "blocks": [("md", (
            "- Non-dividend-paying stock, constant r, F = S e^(rT); European options.\n"
            "- beta = 1 for equities (lognormal backbone); calibration fits alpha, rho, nu for one maturity.\n"
            "- Hagan's formula is an asymptotic expansion (accurate for short/medium maturities and moderate nu).\n"
            "- Monte Carlo prices are computed under the T-forward measure and discounted at r."))]},
    ]


_REGISTRY = {"portfolio": _portfolio, "arbitrage": _arbitrage, "hedging": _hedging, "sabr": _sabr}


@lru_cache(maxsize=None)
def get_help(key: str) -> list:
    return _REGISTRY[key]()


def latex_of(payload) -> str:
    return payload if isinstance(payload, str) else L(payload)
