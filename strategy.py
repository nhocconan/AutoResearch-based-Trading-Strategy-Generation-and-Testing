#!/usr/bin/env python3
# 6h_KAMA_Regime_Trend_Breakout
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing a dynamic trend filter. 
# Combined with 1-day Donchian breakout direction and volume confirmation, this strategy aims to capture 
# strong trending moves while avoiding choppy markets. Designed for 6h timeframe with 1d HTF trend filter.
# Expected trade frequency: 15-30 per year per symbol, targeting 60-120 total trades over 4 years.
# Works in both bull and bear markets by using adaptive trend strength and breakout direction from higher timeframe.

name = "6h_KAMA_Regime_Trend_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # KAMA: Kaufman Adaptive Moving Average
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / (volatility[length-1:] + 1e-10)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_values = np.zeros_like(close)
        kama_values[length-1] = close[length-1]
        for i in range(length, len(close)):
            kama_values[i] = kama_values[i-1] + sc[i] * (close[i] - kama_values[i-1])
        return kama_values

    # Calculate KAMA
    kama_values = kama(close, 10, 2, 30)

    # Get 1-day data for Donchian breakout direction
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period)
    donch_high = np.zeros_like(high_1d)
    donch_low = np.zeros_like(low_1d)
    for i in range(20, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-20:i])
        donch_low[i] = np.min(low_1d[i-20:i])
    
    # Donchian breakout direction: 1 if above upper band, -1 if below lower band, 0 otherwise
    donch_direction = np.zeros_like(high_1d)
    donch_direction[high_1d > donch_high] = 1
    donch_direction[low_1d < donch_low] = -1
    
    # Align Donchian direction to 6t timeframe
    donch_direction_aligned = align_htf_to_ltf(prices, df_1d, donch_direction)

    # Volume filter: >1.8x 20-period average
    vol_avg_20 = np.zeros_like(volume)
    vol_series = pd.Series(volume)
    vol_avg_20[20:] = vol_series.rolling(window=20, min_periods=20).mean().values[20:]
    vol_avg_20[:20] = vol_avg_20[20]  # fill initial values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after sufficient data for indicators
        # Skip if any required value is NaN or not available
        if (np.isnan(kama_values[i]) or np.isnan(donch_direction_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA (uptrend) + bullish Donchian breakout + volume spike
            if (close[i] > kama_values[i] and 
                donch_direction_aligned[i] == 1 and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + bearish Donchian breakout + volume spike
            elif (close[i] < kama_values[i] and 
                  donch_direction_aligned[i] == -1 and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or bearish Donchian breakout
            if (close[i] < kama_values[i] or donch_direction_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or bullish Donchian breakout
            if (close[i] > kama_values[i] or donch_direction_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals