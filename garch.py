import numpy as np

def uncond_var_garch(omega: float, alpha: float, beta: float) -> float:
    """
    Unconditional variance of a GARCH(1,1) model:
        h_t = omega + alpha * e_{t-1}^2 + beta * h_{t-1}

    Returns np.nan if alpha + beta >= 1 (no finite unconditional mean).
    """
    denom = 1.0 - alpha - beta
    return np.nan if denom <= 0 else omega / denom

def uncond_var_gjr(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    ind_mean: float = 0.5,
) -> float:
    """
    Unconditional variance of a GJR-GARCH(1,1,1) model:
        h_t = omega + (alpha + gamma * I_{t-1}) * e_{t-1}^2 + beta * h_{t-1}

    `ind_mean` is the expected value of the indicator I_{t-1}
    (0.5 for a symmetric distribution of returns).

    Returns np.nan if alpha + gamma*ind_mean + beta >= 1.
    """
    denom = 1.0 - alpha - gamma * ind_mean - beta
    return np.nan if denom <= 0 else omega / denom

def excess_kurtosis_gjr(
    alpha: float,
    gamma: float,
    beta: float,
    nu: float,
    ind_mean: float = 0.5,
) -> float:
    """
    Unconditional excess kurtosis of GJR-GARCH(1,1,1) returns with
    Student-t innovations.  Formula from:
        He, Terasvirta & Malmsten (2002)  J. Empirical Finance 9, 457-477.

    Requires nu > 4 and delta < 1 for a finite fourth moment; else np.nan.
    """
    if nu <= 4:
        return np.nan  # fourth moment diverges

    k_z = 3.0 + 6.0 / (nu - 4.0)                 # total kurtosis of z_t
    alpha_bar = alpha + gamma * ind_mean
    lam = alpha_bar + beta
    delta = beta**2 + 2.0 * beta * alpha_bar + k_z * alpha_bar**2

    if delta >= 1.0:
        return np.nan  # fourth moment diverges

    total_kurtosis = k_z * (1.0 - lam**2) / (1.0 - delta)
    return total_kurtosis - 3.0                  # convert to excess kurtosis

def total_kurtosis(alpha_val: float,
                   gamma_val: float,
                   beta_val: float,
                   nu_val: float) -> float:
    """
    Unconditional (total) kurtosis of returns from a
    GJR-GARCH(1,1,1) model with Student-t innovations.
    """
    alpha_bar = alpha_val + 0.5 * gamma_val
    lam       = alpha_bar + beta_val
    k_z       = 3.0 + 6.0 / (nu_val - 4.0)
    delta     = (beta_val ** 2
                 + 2.0 * beta_val * alpha_bar
                 + k_z * alpha_bar ** 2)
    if delta >= 1.0:
        return np.inf
    return k_z * (1.0 - lam ** 2) / (1.0 - delta)

def acf_squared_returns_gjr(alpha_val: float,
                        gamma_val: float,
                        beta_val: float,
                        nu_val: float,
                        lags: range) -> list[float]:
    """
    Theoretical autocorrelation of r_t**2 for a GJR-GARCH(1,1,1).
    Formula from He, Terasvirta, and Malmsten (2002).
    """
    alpha_bar = alpha_val + 0.5 * gamma_val
    lam       = alpha_bar + beta_val
    k_z       = 3.0 + 6.0 / (nu_val - 4.0)
    k_r       = total_kurtosis(alpha_val, gamma_val, beta_val, nu_val)
    rho1      = ((alpha_bar ** 2) * k_z
                 + 2.0 * alpha_bar * beta_val
                 + beta_val ** 2) / (k_r - 1.0)
    return [rho1 * (lam ** (k - 1)) for k in lags]