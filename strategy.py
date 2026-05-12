#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Momentum_VolumeFilter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Combined with RSI momentum (avoiding overbought/oversold extremes) and volume confirmation,
this strategy captures sustained moves while filtering false signals. Works in all regimes by
using adaptive trend + momentum confirmation.
"""

name = "4h_KAMA_Trend_With_RSI_Momentum_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for KAMA (same timeframe, but we'll use it for consistency)
    # Actually, we need higher timeframe trend: use 12h for trend filter
    df_12h = get_htf_data(prices, '12h')

    # Calculate KAMA on 12h close
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_12h, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)

    # RSI(14) on 4h close
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[14] = np.nansum(gain[1:15]) / 14
    avg_loss[14] = np.nansum(loss[1:15]) / 14
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume spike: >1.5x 20-period average
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:10] = np.nan  # Not enough data for full window
    vol_ma[-10:] = np.nan
    # Recalculate properly using pandas for correctness
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA (uptrend) + RSI in momentum zone (40-60) + volume spike
            if (close[i] > kama_aligned[i] and 
                40 <= rsi[i] <= 60 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + RSI in momentum zone (40-60) + volume spike
            elif (close[i] < kama_aligned[i] and 
                  40 <= rsi[i] <= 60 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA (trend change)
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA (trend change)
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals