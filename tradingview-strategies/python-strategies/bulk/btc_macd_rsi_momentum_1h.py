#!/usr/bin/env python3
"""
TradingView bulk conversion: Momentum Strategy (BTC/USDT; 1h) - MACD
Compatibility: partial

Adaptations:
- Uses Pine default parameters only.
- Supports long/short switching with next-bar execution.
- Preserves RSI-based entry/exit gating and MACD-slope turn logic.
"""

import numpy as np
import pandas as pd

name = "btc_macd_rsi_momentum_1h_bulk"
timeframe = "1h"
leverage = 1.0


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=np.float64)
    return series.rolling(window=length, min_periods=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True,
    )


def _hma(series: pd.Series, length: int) -> pd.Series:
    half = max(int(length / 2), 1)
    root = max(int(np.sqrt(length)), 1)
    return _wma(2.0 * _wma(series, half) - _wma(series, length), root)


def _tema(series: pd.Series, length: int) -> pd.Series:
    ema1 = _ema(series, length)
    ema2 = _ema(ema1, length)
    ema3 = _ema(ema2, length)
    return 3.0 * ema1 - 3.0 * ema2 + ema3


def _thma(series: pd.Series, length: int) -> pd.Series:
    h1 = _hma(series, length)
    return 3.0 * h1 - _hma(_hma(h1, length), length)


def _linreg(series: pd.Series, length: int, offset: int) -> pd.Series:
    def calc(window: np.ndarray) -> float:
        x = np.arange(len(window), dtype=np.float64)
        slope, intercept = np.polyfit(x, window, 1)
        return slope * (len(window) - 1 + offset) + intercept
    return series.rolling(window=length, min_periods=length).apply(calc, raw=True)


def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"]
    rsi = _rsi(close, 14)

    fast_ma = _ema(close, 12)
    slow_ma = _ema(close, 26)
    macd = fast_ma - slow_ma

    macd = _thma(macd, 36)
    macd = _linreg(macd, 10, 1)

    macd_delta = macd.diff()
    apertura_long = (macd_delta > 0.0) & (macd_delta.shift(1) <= 0.0) & (rsi < 90.0)
    apertura_short = (macd_delta < 0.0) & (macd_delta.shift(1) >= 0.0) & (rsi > 44.0)

    chiusura_short = (rsi < 44.0) | apertura_long
    chiusura_long = (rsi > 90.0) | apertura_short

    signals = np.zeros(len(prices), dtype=np.float64)
    position = 0.0

    for i in range(len(prices)):
        if position < 0.0 and chiusura_short.iloc[i]:
            position = 0.0
        if position > 0.0 and chiusura_long.iloc[i]:
            position = 0.0

        if position <= 0.0 and apertura_long.iloc[i]:
            position = 1.0
        elif position >= 0.0 and apertura_short.iloc[i]:
            position = -1.0

        signals[i] = position

    return signals
