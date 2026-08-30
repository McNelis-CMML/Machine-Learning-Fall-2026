"""Shared helper routines for the volatility class notebooks.

Keep the economic/statistical formulas visible in the notebook where they are taught.
This module contains only routines that are useful for reusing setup or previously
introduced calculations in later classes.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def load_prices(ticker="^GSPC", start="1960-01-01", end=None,
                interval="1d", auto_adjust=True):
    """Download and clean daily market-price data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, for example:
        "AAPL"   Apple
        "MSFT"   Microsoft
        "NVDA"   Nvidia
        "^GSPC"  S&P 500 index
        "^DJI"   Dow Jones Industrial Average
        "^IXIC"  Nasdaq Composite
        "SPY"    S&P 500 ETF
    start : str
        Starting date in YYYY-MM-DD format.
    end : str or None
        Ending date in YYYY-MM-DD format. If None, download through the latest
        available date.
    interval : str
        Yahoo Finance sampling interval, e.g. "1d", "1wk", or "1mo".
    auto_adjust : bool
        If True, Yahoo Finance adjusts OHLC prices for splits and dividends.

    Returns
    -------
    pandas.DataFrame
        Price data with Log_Close, Log_Open, Log_High, Log_Low, and Return.
    """

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data were downloaded for ticker {ticker!r}. "
            "Check the Yahoo Finance ticker symbol and date range."
        )

    # yfinance may return a two-level column index even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    cols_to_check = ["Open", "High", "Low", "Close"]
    missing = [col for col in cols_to_check if col not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data are missing required columns: {missing}")

    # Log prices require strictly positive values.
    for col in cols_to_check:
        data[col] = data[col].where(data[col] > 0, np.nan)

    data[cols_to_check] = data[cols_to_check].ffill()

    data["Log_Close"] = np.log(data["Close"])
    data["Log_Open"] = np.log(data["Open"])
    data["Log_High"] = np.log(data["High"])
    data["Log_Low"] = np.log(data["Low"])
    data["Return"] = data["Log_Close"].diff()

    data.attrs["ticker"] = ticker
    return data


def load_sp500(start="1960-01-01", end=None):
    """Convenience wrapper for the S&P 500 index."""
    return load_prices("^GSPC", start=start, end=end)


def load_vix(start="1990-01-01", end=None):
    """Download daily VIX data and return a cleaned DataFrame."""
    vix = yf.download(
        "^VIX", start=start, end=end, interval="1d",
        auto_adjust=True, progress=False
    )
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix["VIX"] = vix["Close"]
    return vix


def garman_klass_rolling(spx, window=20):
    """Rolling Garman-Klass range volatility in daily-return units."""
    hl = np.log(spx["High"] / spx["Low"])
    co = np.log(spx["Close"] / spx["Open"])
    gk_var = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    return np.sqrt(gk_var.rolling(window).mean().clip(lower=0)).rename("GK")


def ok_rolling(spx, window=20):
    """Rolling OK range-volatility measure used in the class notebook."""
    hl = np.log(spx["High"] / spx["Low"])
    co = np.log(spx["Close"] / spx["Open"])
    ok_var = 0.811 * hl**2 - 0.369 * co**2
    return np.sqrt(ok_var.rolling(window).mean().clip(lower=0)).rename("OK")


def fit_gjr_garch_in_mean_volatility(spx, nlags=5, disp="off"):
    """Re-estimate the GJR-GARCH-in-Mean model taught in Class 3.

    Returns
    -------
    daily_vol : pandas.Series
        Conditional volatility converted back from percentage units to daily-return units.
    results : arch result object
        Full fitted model results.
    """
    from arch.univariate import ARCHInMean, GARCH, StudentsT

    r = spx["Return"].dropna().to_numpy()
    y = 100 * r[nlags:]

    model = ARCHInMean(
        y,
        lags=nlags,
        volatility=GARCH(p=1, o=1, q=1),
        distribution=StudentsT(),
        rescale=False,
    )

    results = model.fit(
        disp=disp,
        options={"maxiter": 5000, "ftol": 1e-8},
    )

    dates = spx["Return"].dropna().index[nlags:]
    daily_vol = pd.Series(
        results.conditional_volatility / 100.0,
        index=dates,
        name="GJR_GARCH_M",
    )

    return daily_vol, results


def load_fred(series, start, end=None):
    """
    Download a FRED series using the public CSV download.

    Parameters
    ----------
    series : str
        FRED code, e.g. "DFF", "DGS10", "BAA10Y".
    start : str
        Starting date in YYYY-MM-DD format.
    end : str or None
        Optional ending date.

    Returns
    -------
    pandas.DataFrame
        FRED series indexed by date.
    """

    series = str(series).strip().upper()

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

    data = pd.read_csv(url)

    # FRED has used both of these names for the date column
    if "observation_date" in data.columns:
        date_col = "observation_date"
    elif "DATE" in data.columns:
        date_col = "DATE"
    else:
        date_col = data.columns[0]

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce"
    )

    data = data.dropna(subset=[date_col])
    data = data.set_index(date_col)

    data.index.name = "Date"

    # Convert missing observations such as "." to NaN
    data[series] = pd.to_numeric(
        data[series],
        errors="coerce"
    )

    # Select requested dates
    data = data.loc[
        data.index >= pd.to_datetime(start)
    ]

    if end is not None:
        data = data.loc[
            data.index <= pd.to_datetime(end)
        ]

    return data[[series]]
