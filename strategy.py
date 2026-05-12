#!/usr/bin/env python3

# 12h_1D_KAMA_Trend_RSI_Filter
# Hypothesis: KAMA adapts to market efficiency, providing smooth trend that avoids whipsaws in ranging markets.
# Combined with RSI overbought/oversold levels for mean reversion entries in the direction of higher timeframe trend.
# Works in both bull and bear markets by requiring 1d trend alignment and volatility filter.
# Targets 15-25 trades/year on 12h timeframe.

name = "12h_1D_KAMA_Trend_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate KAMA (12-period) on 12h data
    # Efficiency ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, k=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close, k=1)), axis=0)  # sum of absolute changes
    # Handle the first 10 elements where diff doesn't work
    change_full = np.full(n, np.nan)
    volatility_full = np.full(n, np.nan)
    change_full[10:] = change
    # Calculate volatility as rolling sum of |diff|
    diff_abs = np.abs(np.diff(close, k=1))
    diff_abs_full = np.full(n, np.nan)
    diff_abs_full[1:] = diff_abs
    vol_sum = np.full(n, np.nan)
    for i in range(10, n):
        vol_sum[i] = np.sum(diff_abs_full[i-9:i+1])  # sum of last 10 absolute changes
    
    er = np.where(vol_sum > 0, change_full / vol_sum, 0)
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start with close after 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate RSI (14-period) on 12h data
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # insert 0 at beginning
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    # First average: simple mean of first 14 periods
    avg_gain[13] = np.mean(gain[1:14]) if n > 13 else np.nan
    avg_loss[13] = np.mean(loss[1:14]) if n > 13 else np.nan
    
    # Wilder smoothing: subsequent values
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume filter: current volume > 1.3x average of last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price below KAMA (dip in uptrend) with RSI oversold and volume confirmation
            if close[i] < kama[i] and rsi[i] < 30 and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price above KAMA (rally in downtrend) with RSI overbought and volume confirmation
            elif close[i] > kama[i] and rsi[i] > 70 and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above KAMA or RSI overbought
            if close[i] > kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below KAMA or RSI oversold
            if close[i] < kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals