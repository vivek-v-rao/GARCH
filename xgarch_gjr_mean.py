"""Load prices for multiple symbols, fit GARCH / GJR-GARCH to **each**
column, compare ACFs of squared returns and normalised returns, and
(optionally) bin by model-implied volatility.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from pandas_util import read_csv_date_index
from stats import (
    acf,
    print_acf_table,
    print_stats_table,
    plot_norm_kde,
    return_stats,
    vol_bin_stats,
)

# ─────────────── User-set toggles ───────────────
pd.set_option("display.float_format", "{:.4f}".format)
plot_gjr_garch = False # plot time series of standardized returns and conditional volatility
plot_norm_dist = True # distribution of normalized returns, compared with normal
fit_garch = True
fit_gjr_garch = True
max_lag = 10
dist = "skewt"
rf_rate = 0.03
days_year = 252.0
vol_bin_width = 1.0
max_vol_threshold = 2.0
prices_file = "spy_tlt.csv"        # any CSV with one column per symbol
# ────────────────────────────────────────────────


def fit_one_symbol(sym: str, px: pd.Series) -> None:
    """Fit models, print diagnostics, and (optionally) plot for *one* symbol."""
    if px.isna().all():
        print(f"\n{sym}: no data – skipped\n")
        return

    print(f"\nsymbol = {sym.lower()}")
    print("\nfirst and last dates:\n" + px.iloc[[0, -1]].to_string())

    # ---------- raw returns ----------
    xret = 100 * (px.pct_change().dropna() - rf_rate / days_year)
    raw_stats_row = return_stats(xret.to_numpy(), days_year)

    # ---------- containers / flags ----------
    norm_stats_row: list[float] | None = None
    std_ret_series: pd.Series | None = None
    acf_done = False

    # ---------- shared post-processing ----------
    def process_std_ret(std_series: pd.Series) -> None:
        nonlocal norm_stats_row, std_ret_series, acf_done

        if norm_stats_row is None:
            norm_stats_row = return_stats(std_series.to_numpy(), days_year)
        std_ret_series = std_series

        if max_lag > 0 and not acf_done:
            sq_raw = (xret.loc[std_series.index] ** 2).to_numpy()
            sq_std = (std_series**2).to_numpy()
            print_acf_table(
                [acf(sq_raw, k) for k in range(1, max_lag + 1)],
                [acf(sq_std, k) for k in range(1, max_lag + 1)],
                range(1, max_lag + 1),
            )
            acf_done = True

    cond_vol: pd.Series | None = None

    # ----- GARCH(1,1) -----
    if fit_garch:
        res = arch_model(xret, dist=dist).fit(update_freq=0, disp="off")
        print(res.summary(), end="\n\n")
        cond_vol = res.conditional_volatility.dropna()
        process_std_ret(xret.loc[cond_vol.index] / cond_vol)

    # ----- GJR-GARCH(1,1,1) -----
    if fit_gjr_garch:
        res = arch_model(xret, p=1, o=1, q=1, dist=dist).fit(
            update_freq=0, disp="off"
        )
        print(res.summary())
        cond_vol = res.conditional_volatility.dropna()
        process_std_ret(xret.loc[cond_vol.index] / cond_vol)

        if plot_gjr_garch:
            res.plot(annualize="D")
            plt.suptitle(f"{sym}: GJR-GARCH(1,1,1)", y=0.95, fontsize=10)
            plt.show()

    # ---------- final output ----------
    print_stats_table(raw_stats_row, norm_stats_row)

    if plot_norm_dist and std_ret_series is not None:
        plot_norm_kde(std_ret_series, log_ratio=True)

    # ---------- volatility-bin statistics ----------
    if vol_bin_width is not None and cond_vol is not None:
        df_vol_stats = vol_bin_stats(
            xret.loc[cond_vol.index],
            cond_vol,
            vol_bin_width,
            max_vol_threshold,
            days_year,
        )
        print(
            "Volatility-bin statistics:\n"
            + df_vol_stats.to_string(index=False),
            end="\n\n",
        )


def main() -> None:
    """Top-level routine: load data and run `fit_one_symbol` for every column."""
    all_px: pd.DataFrame = read_csv_date_index(prices_file)

    if all_px.empty:
        raise ValueError(f"{prices_file!r} is empty or not found")

    # Preserve original column order; drop all-NaN columns
    for sym, px in all_px.dropna(axis=1, how="all").items():
        try:
            fit_one_symbol(sym, px)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n{sym}: error – {exc}\n")


if __name__ == "__main__":
    main()
