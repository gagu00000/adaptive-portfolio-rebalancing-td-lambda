"""
data_fetcher.py
================
Fetches real market data from Yahoo Finance and prepares
features used as the RL environment's state observations.

Features computed per asset per day:
  - daily log return
  - 10-day rolling volatility (std of log returns)
  - 10-day momentum score (log return over window)
"""

import numpy as np
import pandas as pd
import yfinance as yf


# Default tickers used for the portfolio
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]

# Train/test split date
SPLIT_DATE = "2023-01-01"


def fetch_data(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    print(f"[Data] Downloading data for: {tickers}")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"].dropna()
    print(f"[Data] Downloaded {len(close)} trading days, {len(close.columns)} assets.")
    return close


def compute_features(close: pd.DataFrame, window: int = 10) -> dict:
    """
    Given a DataFrame of adjusted close prices, returns a dict with:
      - 'returns'    : daily log returns (T x N)
      - 'volatility' : rolling std of log returns (T x N)
      - 'momentum'   : rolling sum of log returns (T x N)
      - 'tickers'    : list of asset names
    All DataFrames are aligned and NaN rows dropped.
    """
    log_ret = np.log(close / close.shift(1))
    volatility = log_ret.rolling(window).std()
    momentum = log_ret.rolling(window).sum()

    # Align and drop NaN
    features = pd.concat(
        [log_ret, volatility, momentum], axis=1, keys=["ret", "vol", "mom"]
    ).dropna()

    return {
        "returns": features["ret"],
        "volatility": features["vol"],
        "momentum": features["mom"],
        "tickers": list(close.columns),
        "dates": features.index,
    }


def train_test_split(
    features: dict, split_date: str = SPLIT_DATE
) -> tuple[dict, dict]:
    """Split features into train and test sets by date."""
    dates = features["dates"]
    mask = dates < split_date

    def _split(df, mask):
        return df.loc[mask], df.loc[~mask]

    ret_tr, ret_te = _split(features["returns"], mask)
    vol_tr, vol_te = _split(features["volatility"], mask)
    mom_tr, mom_te = _split(features["momentum"], mask)

    train = {
        "returns": ret_tr,
        "volatility": vol_tr,
        "momentum": mom_tr,
        "tickers": features["tickers"],
        "dates": dates[mask],
    }
    test = {
        "returns": ret_te,
        "volatility": vol_te,
        "momentum": mom_te,
        "tickers": features["tickers"],
        "dates": dates[~mask],
    }
    print(
        f"[Data] Train: {len(train['dates'])} days | Test: {len(test['dates'])} days"
    )
    return train, test


def load_and_prepare(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    split_date: str = SPLIT_DATE,
    window: int = 10,
) -> tuple[dict, dict]:
    """High-level function: fetch → compute features → split."""
    close = fetch_data(tickers, start, end)
    features = compute_features(close, window)
    return train_test_split(features, split_date)


if __name__ == "__main__":
    train, test = load_and_prepare()
    print("Train returns shape:", train["returns"].shape)
    print("Test returns shape:", test["returns"].shape)
