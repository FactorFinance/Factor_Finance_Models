# Factor Finance — Quant Models on Indian Markets

Quantitative finance models built on real NSE/BSE data.
Each model is built to understand the actual math behind
Indian F&O markets — not generic textbook examples.

YouTube: youtube.com/@FactorFinance_2026
---

## Model 1 — SPAN Margin Calculator

**What it does:**
Implements NSE's SPAN margin calculation from scratch.
Runs all 16 price-volatility scenarios, calculates worst-case
portfolio loss, adds exposure margin to get total initial margin.

**Files:**
- `span_margin_model.py` — Full Python implementation
- `SPAN_Margin_Model_FactorFinance_v2.xlsx` — Excel model

**What you can do with it:**
- Calculate SPAN margin for any Nifty futures position
- See how margin changes as India VIX rises
- Compare naked vs hedged position margin requirements
- Understand portfolio margining — why hedging reduces margin

**Key finding:**
Margin is not a fixed deposit. It is a live risk model output
running 16 scenarios every session. When VIX rises before
RBI policy or Budget — every scenario recalculates. Your
margin changes. You changed nothing.

**Video explanation:**
[Why Your F&O Margin Increases Overnight](youtube.com/@FactorFinance_2026)

---

## Coming Next

- Options expiry analysis — do 70% really expire worthless?
- Naked call selling expectancy study — 2021 to 2024 NSE data
- Merton credit model on Indian corporate defaults
- Black-Scholes implementation with Indian market calibration

*Factor Finance — Quant models. Real data. Indian markets.*
```

