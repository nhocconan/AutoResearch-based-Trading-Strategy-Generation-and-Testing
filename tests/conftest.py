#!/usr/bin/env python3
"""
Shared pytest fixtures and synthetic-data builders.

The real market data (data/processed/*.parquet) is gitignored and absent in
CI, so the engine is tested entirely against *synthetic* OHLCV frames. This is
deliberate: the backtest/evaluate/mtf functions all accept DataFrames directly,
so we can prove no-lookahead, cost, and metric behaviour without any download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n: int = 300,
    start: str = "2021-01-01",
    freq: str = "1h",
    close: np.ndarray | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a synthetic OHLCV frame with the engine's expected schema.

    Columns: open_time (datetime64[ns]), open, high, low, close, volume.
    If ``close`` is supplied it is used verbatim; otherwise a gentle random
    walk is generated so high/low straddle open/close sensibly.
    """
    rng = np.random.default_rng(seed)
    open_time = pd.date_range(start=start, periods=n, freq=freq).values

    if close is None:
        steps = rng.normal(0.0, 0.5, size=n)
        close = 100.0 + np.cumsum(steps)
        close = np.maximum(close, 1.0)
    else:
        close = np.asarray(close, dtype=np.float64)
        assert len(close) == n, "close length must equal n"

    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]  # open == previous close, a common convention
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    low = np.maximum(low, 0.5)
    volume = rng.uniform(100.0, 1000.0, size=n)

    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def make_flat_ohlcv(n: int = 100, price: float = 100.0, freq: str = "1h") -> pd.DataFrame:
    """Constant-price frame — isolates costs/funding from market PnL."""
    close = np.full(n, price, dtype=np.float64)
    df = make_ohlcv(n=n, close=close, freq=freq)
    # Force high/low/open onto the flat price so bar return is exactly zero.
    df["open"] = price
    df["high"] = price
    df["low"] = price
    df["close"] = price
    return df


def make_trending_ohlcv(n: int = 200, slope: float = 0.5, freq: str = "1h") -> pd.DataFrame:
    """Monotonically rising frame — long should profit, short should lose."""
    close = 100.0 + slope * np.arange(n, dtype=np.float64)
    return make_ohlcv(n=n, close=close, freq=freq)


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def flat_ohlcv():
    return make_flat_ohlcv()


@pytest.fixture
def trending_ohlcv():
    return make_trending_ohlcv()
