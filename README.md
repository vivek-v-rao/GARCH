# GARCH
Symmetric and asymmetric GARCH models to predict return volatility. The main program is `xgarch_gjr_mean.py`, which
produced the results in `results_spy_tlt.txt`, discussed below.

## Model Comparison

| Asset | Model               | Log-L    | ΔAIC vs GARCH | ΔBIC vs GARCH | γ (leverage) | p-value | Preferred? |
|-------|---------------------|----------|---------------|---------------|--------------|---------|------------|
| **SPY** | GARCH         | −7712.32 | 0             | 0             | —            | —       | — |
|        | **GJR-GARCH** | **−7610.67** | **−201.3**     | **−194.7**     | **+0.190** | < 10⁻¹⁰ | **Yes** |
| **TLT** | **GARCH**     | **−7118.97** | 0             | 0             | —            | —       | **Yes** |
|        | GJR-GARCH (1, 1, 1)   | −7116.66 | −2.6          | +4.0          | −0.015       | 0.083   | No |

---

## Qualitative Differences in Fitted Dynamics

| Feature | SPY (Equities) | TLT (Long-Bond ETF) |
|---------|----------------|---------------------|
| **Leverage effect** | Strong (γ ≈ 0.19) → volatility spikes after negative returns. | Absent; γ insignificant. |
| **Persistence β** | 0.848 → total persistence ≈ 0.98 | 0.942 → very slow-moving vol (≈ 0.995) |
| **Residual ACF of r² (lags 1-10)** | |ρ| ≤ 0.03 ⇒ clustering removed. | Small positive hump remains. |
| **Return by vol-bin** | Mean return rises with vol; Sharpe falls slowly. | Mean return dips in mid-vol, rises in tail (flight-to-quality). |

---

## Practical Take-aways

* **SPY:** Use a leverage-sensitive model (e.g., GJR-GARCH) for risk management and option pricing.  
* **TLT:** Plain GARCH(1, 1) is sufficient; asymmetry adds little value.  
* **Portfolio risk:** Equity tails are far fatter and more asymmetric than Treasury tails—allocate risk accordingly.

A volatility model with asymmetry, such as GJR-GARCH, is needed to
capture the fact that conditional volatility can rise after big
positive returns. On April 9, the S&P 500 rose 9.5% and VIX fell from
52 to 34. The trailing historical volatility jumped that day because
of the huge return. . In the daily_vol_forecasts.csv file you can see
that the GJR-GARCH model fit to SPY predicted a small drop in realized
vol at the close of April 9 vs. April 8, while a symmetric GARCH model
predicted a big increase, the opposite of VIX's behavior.
