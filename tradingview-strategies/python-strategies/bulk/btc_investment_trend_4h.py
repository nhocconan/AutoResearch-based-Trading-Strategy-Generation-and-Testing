#!/usr/bin/env python3
"""
TradingView bulk conversion: Automated Bitcoin (BTC) Investment Strategy
Compatibility: partial

Adaptations:
- Uses Pine default long-only settings.
- Partial take-profits are mapped to fractional target positions.
- Trailing ATR exit is evaluated from bar data and fills next bar in the repo engine.
"""

import numpy as np
import pandas as pd

name = "btc_investment_trend_4h_bulk"
timeframe = "4h"
leverage = 1.0


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _tema(series: pd.Series, length: int) -> pd.Series:
    ema1 = _ema(series, length)
    ema2 = _ema(ema1, length)
    ema3 = _ema(ema2, length)
    return 3.0 * ema1 - 3.0 * ema2 + ema3


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) == 0:
        return pd.Series(out, index=close.index)
    out[0] = tr.iloc[0]
    alpha = 1.0 / length
    for i in range(1, len(close)):
        out[i] = out[i - 1] + alpha * (tr.iloc[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def _cross_over(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"]
    high = prices["high"]
    low = prices["low"]

    lead_line1 = _tema(close, 25)
    lead_line2 = close.rolling(window=100, min_periods=100).apply(
        lambda x: np.polyfit(np.arange(len(x), dtype=np.float64), x, 1)[0] * (len(x) - 1) + np.polyfit(np.arange(len(x), dtype=np.float64), x, 1)[1],
        raw=True,
    )
    atr = _atr(high, low, close, 8)
    trail1 = pd.Series(np.nan, index=prices.index, dtype=np.float64)
    for i in range(len(prices)):
        sc = close.iloc[i]
        sl1 = 3.5 * atr.iloc[i]
        prev = trail1.iloc[i - 1] if i > 0 else np.nan
        iff_1 = sc - sl1 if sc > (prev if np.isfinite(prev) else 0.0) else sc + sl1
        if i > 0 and sc < (prev if np.isfinite(prev) else 0.0) and close.iloc[i - 1] < (prev if np.isfinite(prev) else 0.0):
            trail1.iloc[i] = min(prev, sc + sl1)
        else:
            trail1.iloc[i] = iff_1
    trail1_high = trail1.rolling(window=50, min_periods=1).max()

    entry_long = _cross_over(lead_line1, lead_line2) & (trail1_high < close)

    signals = np.zeros(len(prices), dtype=np.float64)
    position = 0.0
    entry_price = np.nan
    tp1_done = False
    tp2_done = False

    for i in range(len(prices)):
        if position > 0.0:
            long_sl_input_level = entry_price * (1.0 - 0.05)
            exit_long = (
                (close.iloc[i] < trail1_high.iloc[i])
                or _cross_over(lead_line2, lead_line1).iloc[i]
                or (close.iloc[i] < long_sl_input_level)
            )
            tp1 = entry_price * 1.15
            tp2 = entry_price * 1.30
            if (not tp1_done) and high.iloc[i] >= tp1:
                position = min(position, 0.8)
                tp1_done = True
            if (not tp2_done) and high.iloc[i] >= tp2:
                position = min(position, 0.6)
                tp2_done = True
            if exit_long:
                position = 0.0
                entry_price = np.nan
                tp1_done = False
                tp2_done = False

        if position == 0.0 and entry_long.iloc[i]:
            position = 1.0
            entry_price = close.iloc[i]
            tp1_done = False
            tp2_done = False

        signals[i] = position

    return signals
