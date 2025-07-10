"""
Load daily prices for multiple symbols, compute daily returns, fit both
GARCH(1,1) and GJR-GARCH(1,1,1) models to each return series, and collect
the price levels, returns, and annualised volatility forecasts in one
DataFrame.

Per symbol the output columns appear in this order:
    <sym>_px   <sym>_ret   <sym>_garch   <sym>_gjr
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from garch import uncond_var_garch, uncond_var_gjr

from pandas_util import read_csv_date_index
from stats import (
    acf,
    print_acf_table,
    print_stats_table,
    plot_norm_kde,
    return_stats,
    vol_bin_stats,
)

# ------------------- user-set toggles -------------------
pd.set_option("display.float_format", "{:.4f}".format)
plot_gjr_garch = False # plot standardized residuals and conditional volatility
plot_norm_dist = False
fit_garch = True # fit symmetric GARCH model
fit_gjr_garch = True # fit asymmetric GJR-GARCH model
nacf = 10 # of autocorrelations of squared returns to print
dist = "normal"  # conditional distribution -- "normal", "t", "skewt", "ged"
rf_rate = 0.03  # risk-free interest rate
days_year = 252.0  # trading days per year
ann_factor = np.sqrt(days_year)
vol_bin_width = 1.0 * ann_factor
max_vol_threshold = 2.0 * ann_factor
use_log_returns = True # True/False gives log/proportional returns
prices_file = "spy_tlt.csv" 
vol_forecast_file = "daily_vol_forecasts.csv"
ret_scale = 100 # scaling of returns
max_assets = None  # Set to an integer like 5 to limit, or None for no limit
print_first_last_vol = False # print few few and last few vol predictions
# --------------------------------------------------------


def fit_one_symbol(sym: str, px: pd.Series) -> pd.DataFrame:
    """Return DataFrame with price, return and volatility columns for one symbol."""
    if px.isna().all():
        print(f"\n{sym}: no data - skipped\n")
        return pd.DataFrame()

    print(f"\nsymbol = {sym.lower()}")
    print("\nfirst and last dates:\n" + px.iloc[[0, -1]].to_string())

    # -------- returns --------
    if use_log_returns:
        xret = ret_scale * np.log(px.clip(lower=0)).diff().dropna()
    else:
        xret = ret_scale * px.pct_change().dropna()
    xret = xret - rf_rate / days_year
    raw_stats_row = return_stats(xret.to_numpy(), days_year)

    # build output columns
    cols: list[pd.Series] = [
        px.rename(f"{sym}_px"),
        xret.reindex(px.index).rename(f"{sym}_ret"),
    ]

    norm_stats_row: list[float] | None = None
    std_ret_series: pd.Series | None = None
    acf_done = False
    last_vol: list[pd.Series | None] = [None]   # mutable wrapper

    # helper to add volatility iff non-empty
    def _add_vol_col(vol_ser: pd.Series, tag: str) -> None:
        if vol_ser.empty:
            print(f"note: {sym}_{tag} volatility empty - skipped")
            return
        cols.append(vol_ser.rename(f"{sym}_{tag}"))
        last_vol[0] = vol_ser

    # helper to process standardised returns
    def process_std_ret(std_series: pd.Series) -> None:
        nonlocal norm_stats_row, std_ret_series, acf_done
        if std_series.empty:
            return
        if norm_stats_row is None:
            norm_stats_row = return_stats(std_series.to_numpy(), days_year)
        std_ret_series = (std_series - np.mean(std_series))/np.std(std_series)
        if nacf > 0 and not acf_done:
            sq_raw = (xret.loc[std_series.index] ** 2).to_numpy()
            sq_std = (std_series ** 2).to_numpy()
            print_acf_table(
                [acf(sq_raw, k) for k in range(1, nacf + 1)],
                [acf(sq_std, k) for k in range(1, nacf + 1)],
                range(1, nacf + 1),
            )
            acf_done = True

    ret_vol = np.sqrt(days_year) * xret.std()
    # ----- GARCH(1,1) -----
    if fit_garch:
        res = arch_model(xret, dist=dist).fit(update_freq=0, disp="off")
        print(res.summary(), end="\n\n")
        p = res.params
        uc_vol = np.sqrt(days_year * uncond_var_garch(p["omega"], p["alpha[1]"],
            p["beta[1]"]))
        print("unconditional, empirical daily vol:", "%8.4f" % uc_vol, "%8.4f" % ret_vol, end="\n\n")
        vol = np.sqrt(days_year) * res.conditional_volatility.dropna()
        _add_vol_col(vol, "garch")
        process_std_ret(
            xret.loc[vol.index] / vol if not vol.empty else pd.Series(dtype=float)
        )

    # ----- GJR-GARCH(1,1,1) -----
    if fit_gjr_garch:
        res = arch_model(xret, p=1, o=1, q=1, dist=dist).fit(update_freq=0, disp="off")
        print(res.summary())
        p = res.params
        uc_vol = np.sqrt(days_year * uncond_var_gjr(p["omega"], p["alpha[1]"],
            p["gamma[1]"], p["beta[1]"]))
        print("unconditional, empirical daily vol:", "%8.4f" % uc_vol, "%8.4f" % ret_vol, end="\n\n")
        vol = np.sqrt(days_year) * res.conditional_volatility.dropna()
        _add_vol_col(vol, "gjr")
        process_std_ret(
            xret.loc[vol.index] / vol if not vol.empty else pd.Series(dtype=float)
        )
        if plot_gjr_garch and not vol.empty:
            res.plot(annualize="D")
            plt.suptitle(f"{sym}: GJR-GARCH", y=0.95, fontsize=10)
            plt.show()

    # -------- summary tables --------
    print_stats_table(raw_stats_row, norm_stats_row)
    if plot_norm_dist and std_ret_series is not None:
        plot_norm_kde(std_ret_series, log_ratio=True,
            title_prefix=sym + " GJR-GARCH ")

    # -------- vol-bin stats (robust) --------
    if (
        vol_bin_width is not None
        and last_vol[0] is not None
        and not last_vol[0].empty
    ):
        try:
            df_vol_stats = vol_bin_stats(
                xret.loc[last_vol[0].index],
                last_vol[0],
                vol_bin_width,
                max_vol_threshold,
                days_year,
            )
            print("Volatility-bin statistics:\n" + df_vol_stats.to_string(index=False))
        except Exception as exc:
            print(f"note: could not compute vol-bin stats for {sym}: {exc}")

    return pd.concat(cols, axis=1)


def main() -> pd.DataFrame | None:
    """Run pipeline for every symbol and return combined DataFrame."""
    all_px: pd.DataFrame = read_csv_date_index(prices_file)
    if all_px.empty:
        raise ValueError(f"{prices_file!r} is empty or not found")

    frames: list[pd.DataFrame] = []

    for i, (sym, px) in enumerate(all_px.dropna(axis=1, how="all").items()):
        if max_assets is not None and i >= max_assets:
            break
        try:
            out_df = fit_one_symbol(sym, px)
            if not out_df.empty:
                frames.append(out_df)
        except Exception as exc:
            print(f"\n{sym}: error - {exc}\n")

    if not frames:
        print("\nNo forecasts were generated.")
        return None

    df_final = pd.concat(frames, axis=1).sort_index()
    if print_first_last_vol:
        print("\nCombined prices, returns, and volatility forecasts:")
        print("\ninitial:\n" + df_final.head().to_string())
        print("\nfinal:\n" + df_final.tail().to_string())
    if vol_forecast_file is not None:
        df_final.to_csv(vol_forecast_file)
        print("\nwrote vols to", vol_forecast_file)
    return df_final


if __name__ == "__main__":
    main()
