"""Arbitrage detection with synthetic options built from stock + bonds (Module 2).

Three independent checks:
1. Put-call parity (model-free):   C - P = S - K e^{-rT}
2. No-arbitrage bounds (model-free):
       max(0, S - K e^{-rT}) <= C <= S
       max(0, K e^{-rT} - S) <= P <= K e^{-rT}
3. Black-Scholes replication (model-dependent):
       C = N(d1) S - K e^{-rT} N(d2)       (Delta shares + borrowing)
       P = -N(-d1) S + K e^{-rT} N(-d2)    (short shares + lending)

A strategy is a list of *legs*. Each leg knows its cash flow today and its
payoff at maturity, so every strategy can be verified numerically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import bs


@dataclass
class Leg:
    kind: str          # 'call', 'put', 'stock', 'bond'
    sign: int          # +1 long / -1 short
    qty: float         # units (> 0)
    price: float       # unit price today
    K: float = 0.0     # strike (options) or face value (bond)

    @property
    def action(self) -> str:
        if self.kind == "bond":
            return "Lend (buy zero-coupon bond)" if self.sign > 0 else "Borrow (sell zero-coupon bond)"
        if self.kind == "stock":
            return "Buy" if self.sign > 0 else "Short-sell"
        return "Buy" if self.sign > 0 else "Sell (write)"

    @property
    def instrument(self) -> str:
        if self.kind == "call":
            return f"European call, K = {self.K:,.2f}"
        if self.kind == "put":
            return f"European put, K = {self.K:,.2f}"
        if self.kind == "stock":
            return "Underlying stock"
        return f"Zero-coupon bond paying {self.K:,.2f} at T"

    @property
    def cash_t0(self) -> float:
        return -self.sign * self.qty * self.price

    def payoff(self, ST):
        ST = np.asarray(ST, dtype=float)
        if self.kind == "call":
            u = np.maximum(ST - self.K, 0.0)
        elif self.kind == "put":
            u = np.maximum(self.K - ST, 0.0)
        elif self.kind == "stock":
            u = ST
        else:
            u = np.full_like(ST, self.K)
        return self.sign * self.qty * u


@dataclass
class Strategy:
    title: str
    verdict: str
    legs: list = field(default_factory=list)
    maturity_note: str = ""
    model_free: bool = True

    @property
    def net_cash_t0(self) -> float:
        return float(sum(l.cash_t0 for l in self.legs))

    def payoff(self, ST):
        ST = np.asarray(ST, dtype=float)
        tot = np.zeros_like(ST)
        for l in self.legs:
            tot = tot + l.payoff(ST)
        return tot

    def steps(self) -> list[str]:
        out = []
        for i, l in enumerate(self.legs, 1):
            cf = l.cash_t0
            unit = "shares" if l.kind == "stock" else ("bonds" if l.kind == "bond" else "options")
            out.append(
                f"Step {i}: {l.action} {l.qty:,.4g} {unit} - {l.instrument} at {l.price:,.4f} "
                f"-> cash flow today {cf:+,.2f}"
            )
        return out

    def legs_table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Step": i, "Action": l.action, "Instrument": l.instrument, "Quantity": l.qty,
            "Unit price": l.price, "Cash flow at t=0": l.cash_t0,
        } for i, l in enumerate(self.legs, 1)])

    def scenario_table(self, K: float, T: float, r: float) -> pd.DataFrame:
        pts = sorted({round(K * f, 4) for f in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)})
        rows = []
        for st in pts:
            row = {"S_T": st}
            for i, l in enumerate(self.legs, 1):
                row[f"Leg {i} payoff"] = float(l.payoff(st))
            pay = float(self.payoff(st))
            row["Net payoff at T"] = pay
            row["Total P&L at T (incl. t=0 cash invested at r)"] = pay + self.net_cash_t0 * np.exp(r * T)
            rows.append(row)
        return pd.DataFrame(rows)


def pv_strike(K, T, r):
    return K * np.exp(-r * T)


# ---------------------------------------------------------------------------
# 1. Put-call parity
# ---------------------------------------------------------------------------
def parity_analysis(S, K, T, r, C, P, n=1.0, threshold=0.0) -> dict:
    pvK = pv_strike(K, T, r)
    syn_call = P + S - pvK
    syn_put = C - S + pvK
    gap = C - syn_call  # = C - P - S + PV(K)
    res = {"pvK": pvK, "synthetic_call": syn_call, "synthetic_put": syn_put, "gap": gap,
           "call_mispricing": C - syn_call, "put_mispricing": P - syn_put}
    if gap > threshold:
        strat = Strategy(
            title="Call overpriced / put underpriced -> sell the call, buy the synthetic call",
            verdict=f"C = {C:.4f} > P + S - K e^(-rT) = {syn_call:.4f}  (gap {gap:+.4f} per option)",
            legs=[Leg("call", -1, n, C, K), Leg("put", +1, n, P, K),
                  Leg("stock", +1, n, S), Leg("bond", -1, n, pvK, K)],
            maturity_note=(
                "At T: if S_T > K the call is exercised against you: deliver the share you hold and receive K, "
                "which repays the loan (K). If S_T <= K exercise your put: sell the share for K and repay the loan. "
                "In every scenario the net payoff is 0, so the cash collected today is a riskless profit."),
        )
    elif gap < -threshold:
        strat = Strategy(
            title="Call underpriced / put overpriced -> buy the call, sell the synthetic call",
            verdict=f"C = {C:.4f} < P + S - K e^(-rT) = {syn_call:.4f}  (gap {gap:+.4f} per option)",
            legs=[Leg("call", +1, n, C, K), Leg("put", -1, n, P, K),
                  Leg("stock", -1, n, S), Leg("bond", +1, n, pvK, K)],
            maturity_note=(
                "At T: the bond pays K. If S_T > K exercise your call: pay K, receive a share and return it to "
                "close the short sale. If S_T <= K the put is exercised against you: pay K, receive a share and "
                "close the short. Net payoff 0 in all states, the cash collected today is a riskless profit."),
        )
    else:
        strat = None
    res["strategy"] = strat
    res["arbitrage"] = strat is not None
    return res


# ---------------------------------------------------------------------------
# 2. Bounds
# ---------------------------------------------------------------------------
def bounds_analysis(S, K, T, r, C=None, P=None, n=1.0, threshold=0.0) -> dict:
    pvK = pv_strike(K, T, r)
    out = {"bounds": [], "strategies": []}
    if C is not None:
        lo, hi = max(0.0, S - pvK), S
        out["bounds"].append({"Option": "Call", "Lower bound": lo, "Market price": C, "Upper bound": hi,
                              "Status": "OK" if lo - threshold <= C <= hi + threshold else "VIOLATED"})
        if C < lo - threshold:
            out["strategies"].append(Strategy(
                title="Call below its lower bound -> buy the call, short the stock, lend PV(K)",
                verdict=f"C = {C:.4f} < S - K e^(-rT) = {lo:.4f}",
                legs=[Leg("call", +1, n, C, K), Leg("stock", -1, n, S), Leg("bond", +1, n, pvK, K)],
                maturity_note="At T the payoff is max(S_T-K,0) - S_T + K = max(K - S_T, 0) >= 0: "
                              "positive cash today and a non-negative payoff later."))
        if C > hi + threshold:
            out["strategies"].append(Strategy(
                title="Call above the stock price -> sell the call, buy the stock (covered call)",
                verdict=f"C = {C:.4f} > S = {hi:.4f}",
                legs=[Leg("call", -1, n, C, K), Leg("stock", +1, n, S)],
                maturity_note="At T the payoff is S_T - max(S_T-K,0) = min(S_T, K) >= 0."))
    if P is not None:
        lo, hi = max(0.0, pvK - S), pvK
        out["bounds"].append({"Option": "Put", "Lower bound": lo, "Market price": P, "Upper bound": hi,
                              "Status": "OK" if lo - threshold <= P <= hi + threshold else "VIOLATED"})
        if P < lo - threshold:
            out["strategies"].append(Strategy(
                title="Put below its lower bound -> buy the put, buy the stock, borrow PV(K)",
                verdict=f"P = {P:.4f} < K e^(-rT) - S = {lo:.4f}",
                legs=[Leg("put", +1, n, P, K), Leg("stock", +1, n, S), Leg("bond", -1, n, pvK, K)],
                maturity_note="At T the payoff is max(K-S_T,0) + S_T - K = max(S_T - K, 0) >= 0."))
        if P > hi + threshold:
            out["strategies"].append(Strategy(
                title="Put above PV(K) -> sell the put, lend PV(K)",
                verdict=f"P = {P:.4f} > K e^(-rT) = {hi:.4f}",
                legs=[Leg("put", -1, n, P, K), Leg("bond", +1, n, pvK, K)],
                maturity_note="At T the payoff is K - max(K-S_T,0) = min(S_T, K) >= 0."))
    return out


# ---------------------------------------------------------------------------
# 3. Model (Black-Scholes) replication
# ---------------------------------------------------------------------------
def model_analysis(S, K, T, r, sigma, market_price, option="call", n=1.0, threshold=0.0) -> dict:
    model = bs.price(S, K, T, r, sigma, option)
    rep = bs.replicating_portfolio(S, K, T, r, sigma, option)
    iv = bs.implied_vol(market_price, S, K, T, r, option)
    mis = market_price - model
    is_call = str(option).lower().startswith("c")
    res = {"model_price": model, "mispricing": mis, "implied_vol": iv, "replication": rep}
    shares, bond = rep["shares"], rep["bond"]
    kind = "call" if is_call else "put"
    if mis > threshold:
        legs = [Leg(kind, -1, n, market_price, K),
                Leg("stock", +1 if shares > 0 else -1, n * abs(shares), S),
                Leg("bond", +1 if bond > 0 else -1, n * abs(bond) / pv_strike(K, T, r), pv_strike(K, T, r), K)]
        strat = Strategy(
            title=f"{kind.capitalize()} overpriced vs Black-Scholes -> sell it and buy the replicating portfolio",
            verdict=f"Market {market_price:.4f} > model {model:.4f}  (implied vol {iv:.2%} vs model {sigma:.2%})",
            legs=legs, model_free=False,
            maturity_note=("The replicating portfolio must be re-balanced (delta hedge) until T. If realised "
                           "volatility equals the model volatility, the hedge pays the option payoff and you keep "
                           f"about {mis * n:,.2f} (x e^(rT)). This is model-dependent: it is not a riskless arbitrage."))
    elif mis < -threshold:
        legs = [Leg(kind, +1, n, market_price, K),
                Leg("stock", -1 if shares > 0 else +1, n * abs(shares), S),
                Leg("bond", -1 if bond > 0 else +1, n * abs(bond) / pv_strike(K, T, r), pv_strike(K, T, r), K)]
        strat = Strategy(
            title=f"{kind.capitalize()} underpriced vs Black-Scholes -> buy it and sell the replicating portfolio",
            verdict=f"Market {market_price:.4f} < model {model:.4f}  (implied vol {iv:.2%} vs model {sigma:.2%})",
            legs=legs, model_free=False,
            maturity_note=("Re-balance the short replicating portfolio (delta hedge) until T. If realised volatility "
                           f"equals the model volatility you keep about {-mis * n:,.2f} (x e^(rT)). Model-dependent."))
    else:
        strat = None
    res["strategy"] = strat
    res["arbitrage"] = strat is not None
    return res
