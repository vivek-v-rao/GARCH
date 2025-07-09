"""
simulate_and_fit_gjr_t.py
--------------------------------------------------------------------
This script

- simulates returns from a GJR-GARCH(1,1,1) model with Student-t errors,
- fits the same model to the simulated returns,
- prints
    * a table comparing true versus estimated parameters,
    * summary statistics for the true conditional standard deviation
      (sigma_true), the fitted conditional standard deviation
      (sigma_fit), and their difference,
    * statistics for returns (mean, sd, kurtosis) — empirical and theoretical,
    * the correlation between sigma_true and sigma_fit,
    * a DataFrame with the autocorrelation function (ACF) of the
      squared returns at lags 1–20:
        - empirical ACF of r**2,
        - theoretical ACF implied by the true parameters,
        - theoretical ACF implied by the fitted parameters.
--------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, StudentsT, arch_model
from garch import uncond_var_gjr, excess_kurtosis_gjr, acf_squared_returns_gjr
from stats import series_stats

pd.set_option("display.float_format", "{:.6f}".format)

# ------------------------------------------------------------------
# settings
# ------------------------------------------------------------------
np.random.seed(12345)
pd.set_option("display.float_format", "{:.4f}".format)

nobs  = 10**5    # number of observations kept
burn  =  500     # burn-in observations discarded
nacf_sq = 5

mu     = 0.0001
omega  = 0.10
alpha  = 0.05
gamma  = 0.10
beta   = 0.85
nu     = 8.0

theta = np.array([mu, omega, alpha, gamma, beta, nu], dtype=float)

# ------------------------------------------------------------------
# simulate the GJR-GARCH series
# ------------------------------------------------------------------
sim_model = ConstantMean(None)
sim_model.volatility   = GARCH(p=1, o=1, q=1)
sim_model.distribution = StudentsT()

sim = sim_model.simulate(theta, nobs=nobs + burn, burn=burn)
rets       = sim["data"]         # simulated returns r_t
sigma_true = sim["volatility"]   # true conditional sigma_t

# ------------------------------------------------------------------
# fit the same model to the simulated returns
# ------------------------------------------------------------------
fit_model = arch_model(
    rets,
    mean="constant",
    vol="GARCH",
    p=1,
    o=1,
    q=1,
    dist="t",
)
res = fit_model.fit(disp="off")
sigma_fit = res.conditional_volatility  # fitted sigma_t

# ------------------------------------------------------------------
# summary statistics for sigma_t
# ------------------------------------------------------------------
sigma_stats_df = pd.DataFrame.from_dict(
    {
        "sigma_true":  series_stats(sigma_true),
        "sigma_fit":   series_stats(sigma_fit),
        "sigma_diff":  series_stats(sigma_true - sigma_fit),
    },
    orient="index",
)

# ------------------------------------------------------------------
# parameter comparison table
# ------------------------------------------------------------------
true_param_series = pd.Series(
    {
        "mu": mu,
        "omega": omega,
        "alpha[1]": alpha,
        "gamma[1]": gamma,
        "beta[1]": beta,
        "nu": nu,
    }
)
param_df = pd.concat([true_param_series, res.params], axis=1)
param_df.columns = ["true", "estimated"]
param_df["diff"] = param_df["true"] - param_df["estimated"]

# ------------------------------------------------------------------
# correlation between true and fitted sigma_t
# ------------------------------------------------------------------
sigma_corr = sigma_true.corr(sigma_fit)

# ------------------------------------------------------------------
# statistics for returns: empirical and theoretical
# ------------------------------------------------------------------
# extract fitted parameters
mu_hat     = res.params["mu"]
omega_hat  = res.params["omega"]
alpha_hat  = res.params["alpha[1]"]
gamma_hat  = res.params["gamma[1]"]
beta_hat   = res.params["beta[1]"]
nu_hat     = res.params["nu"]

# empirical sample stats
ret_mean_emp  = rets.mean()
ret_std_emp   = rets.std(ddof=0)
ret_kurt_emp  = rets.kurt()  # excess kurtosis

# theoretical stats for true process
ret_std_true  = np.sqrt(uncond_var_gjr(omega, alpha, gamma, beta))
ret_kurt_true = excess_kurtosis_gjr(alpha, gamma, beta, nu)

# theoretical stats for fitted process
ret_std_fit   = np.sqrt(uncond_var_gjr(omega_hat, alpha_hat, gamma_hat, beta_hat))
ret_kurt_fit  = excess_kurtosis_gjr(alpha_hat, gamma_hat, beta_hat, nu_hat)

returns_stats_df = pd.DataFrame(
    {
        "mean":     [ret_mean_emp,  mu,        mu_hat],
        "sd":       [ret_std_emp,   ret_std_true,  ret_std_fit],
        "kurtosis": [ret_kurt_emp,  ret_kurt_true, ret_kurt_fit],
    },
    index=["empirical", "true", "fitted"],
)

if nacf_sq > 0:
    # ------------------------------------------------------------------
    # empirical and theoretical ACF of squared returns
    # ------------------------------------------------------------------
    lags = range(1, nacf_sq)

    emp_acf = [(rets ** 2).autocorr(lag) for lag in lags]

    theo_acf_true = acf_squared_returns_gjr(alpha, gamma, beta, nu, lags)
    theo_acf_fit  = acf_squared_returns_gjr(alpha_hat, gamma_hat, beta_hat, nu_hat, lags)

    acf_df = pd.DataFrame(
        {
            "empirical":            emp_acf,
            "theoretical_true":     theo_acf_true,
            "theoretical_fitted":   theo_acf_fit,
        },
        index=pd.Index(lags, name="lag"),
    )

    print("\nACF of squared returns (empirical vs. theoretical):")
    print(acf_df)

# ------------------------------------------------------------------
# output
# ------------------------------------------------------------------
print("# observations kept:", nobs)
print("\nParameter comparison:")
print(param_df)

print("\nSummary statistics for sigma_t:")
print(sigma_stats_df.to_string())

print(f"\nCorrelation between sigma_true and sigma_fit: {sigma_corr:.6f}")

print("\nStatistics of returns (mean, sd, excess kurtosis):")
print(returns_stats_df)
