#!/usr/bin/env python3
"""
1d_KAMA_Trend_VolumeSpike
Hypothesis: Uses daily timeframe with KAMA (Kaufman Adaptive Moving Average) to capture trend direction,
filtered by volume spike after low volatility regime. Enters long when price > KAMA and volume spikes,
short when price < KAMA and volume spikes. Uses volume contraction/expansion as regime filter to avoid
choppy markets. Designed for low frequency (target 10-25 trades/year) to minimize fee drag and work
in both bull and bear markets by following adaptive trend.
"""

name = "1d_KAMA_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average) - 14-period
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of abs changes over 10 periods
    # Fix dimensions: change starts at index 10, volatility needs same length
    change = change[10:]  # align to close[10:]
    volatility = volatility[10:]  # align
    
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants: fastest SC = 2/(2+1)=0.67, slowest SC = 2/(30+1)=0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    
    # Initialize KAMA array
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9 (10th element)
    
    # Calculate KAMA iteratively (requires loop, but only once outside signal loop)
    for i in range(10, len(close)):
        if not np.isnan(sc[i-10]):  # sc index aligned
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume indicators: 20-period average and volatility regime
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_std_20 = pd.Series(volume).rolling(window=20, min_periods=20).std().values
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = vol_std_20 < (vol_avg_50 * 0.5)  # volatility less than half of 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start from 50 to have enough data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(vol_std_20[i]) or np.isnan(vol_avg_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + low vol regime + volume spike (2x avg)
            if (close[i] > kama[i] and 
                low_vol_regime[i] and 
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + low vol regime + volume spike (2x avg)
            elif (close[i] < kama[i] and 
                  low_vol_regime[i] and 
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals