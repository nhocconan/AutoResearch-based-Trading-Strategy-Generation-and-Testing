#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (1w EMA50) and volume confirmation
# Uses 1w EMA50 for higher timeframe trend alignment (stable across bull/bear regimes)
# Donchian(20) from prior 6h session provides clear breakout levels
# Volume confirmation (>2.0x 50 EMA) filters low-participation false breakouts
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# Works in both bull and bear: weekly trend filter adapts to higher timeframe direction.

name = "6h_Donchian20_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe (completed 1w bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian(20) from prior 6h bar
    # We need to look back 20 periods on the 6h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 50-period EMA of volume on 6h timeframe
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian(20) high + uptrend + volume spike
            if close[i] > highest_20[i] and close[i] > ema50_aligned[i] and volume[i] > (2.0 * vol_ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian(20) low + downtrend + volume spike
            elif close[i] < lowest_20[i] and close[i] < ema50_aligned[i] and volume[i] > (2.0 * vol_ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel OR trend changes
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if close[i] < midpoint or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel OR trend changes
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if close[i] > midpoint or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals