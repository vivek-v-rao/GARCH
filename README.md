# GARCH
Symmetric and asymmetric GARCH models to predict return volatility. The main program is `xgarch_gjr_mean.py`, which
produced the results in `results_spy_tlt.txt`, discussed below.

### 1  Which specification is *best* for each asset?

| Asset   | Model                  | Log-L         | ΔAIC vs GARCH | ΔBIC vs GARCH | γ (leverage) | *p*-value   | Verdict                                         |
| ------- | ---------------------- | ------------- | ------------- | ------------- | ------------ | ----------- | ----------------------------------------------- |
| **SPY** | GARCH(1, 1)            | −7 543.07     | 0             | 0             | —            | —           | ─                                               |
|         | **GJR-GARCH(1, 1, 1)** | **−7 427.64** | **−228.8**    | **−222.2**    | **+0.237**   | **< 10⁻¹⁸** | **Clearly preferred**                           |
| **TLT** | **GARCH(1, 1)**        | −7 082.89     | 0             | **0**         | —            | —           | **Preferred (BIC)**                             |
|         | GJR-GARCH(1, 1, 1)     | −7 081.41     | −1.0          | +5.7          | −0.013       | 0.093       | Tie on AIC; penalised by BIC; γ not significant |

* **SPY:** GJR-GARCH cuts AIC/BIC by ≈ 230 points and has a highly significant leverage term → it is the clear winner.
* **TLT:** The GJR version barely improves the likelihood; with the extra parameter, BIC worsens and γ is insignificant → stick with vanilla GARCH.

---

### 2  Qualitative differences in the fitted dynamics

| Feature                                    | **SPY (equities)**                                                                                      | **TLT (long-bond ETF)**                                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Asymmetric (“leverage”) effect**         | Strong: γ ≈ 0.24 (significant). Volatility reacts much more to *negative* returns.                      | None: γ insignificant → symmetric response to ± shocks.                                                    |
| **Shock–persistence mix**                  | β ≈ 0.87; effective persistence ≈ 0.98 once leverage is counted. Sharp jump after bad news, slow decay. | β ≈ 0.95 and α ≈ 0.05 → persistence ≈ 0.995, driven by high β, not leverage. Volatility changes gradually. |
| **Tail thickness (η)**                     | η ≈ 7 → noticeably fat-tailed.                                                                          | η ≈ 14 → tails only slightly heavier than Gaussian.                                                        |
| **Return skew (λ)**                        | λ ≈ −0.15 → clear negative skew.                                                                        | λ ≈ −0.06 → mild skew only.                                                                                |
| **Vol-return relation**                    | Mean return **rises** with higher vol (risk premium strongest in choppy markets).                       | Mean return **falls** going from 0–1 σ to 1–2 σ bins (typical for “flight-to-quality”).                    |
| **Squared-return ACF after normalisation** | Near-zero → model captures clustering well.                                                             | Residual ACF ≈ 0.02–0.03 → symmetric GARCH leaves a small hump, but much reduced.                          |

---

### 3  Interpretation

* **SPY** behaves like a textbook equity index: downside moves (“leverage”) dominate volatility dynamics. A leverage-aware model (GJR- or EGARCH) is essential.
* **TLT** looks like a symmetric, highly persistent interest-rate asset. GARCH(1, 1) with a high β captures its volatility; leverage terms add little.

---

### 4  Practical takeaway

* **Equities:** Use leverage-sensitive models to avoid understating risk after sell-offs.
* **Treasuries:** A simple GARCH often suffices; volatility is symmetric and slow-moving.
* **Risk management:** Tail parameters differ markedly—fat-tail corrections matter far more for equity than for Treasury-bond portfolios.

