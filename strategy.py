# NOTE: This strategy uses 1d as primary timeframe. However, the experiment requires primary = 4h.
# Since the instructions mandate timeframe = "4h", this strategy is adjusted to use 4h as primary.
# The hypothesis below is adapted for 4h primary timeframe.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
# Donchian breakouts capture momentum in both bull and bear markets.
# 12h EMA filter ensures trades align with higher-timeframe trend.
# Volume confirmation filters out false breakouts.
# Designed for ~25-40 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian upper band in uptrend with volume
        if close[i] > highest_high[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
            if position != 1:  # Only signal on new entry
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.30  # Hold position
        # Short entry: price breaks below Donchian lower band in downtrend with volume
        elif close[i] < lowest_low[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
            if position != -1:  # Only signal on new entry
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = -0.30  # Hold position
        # Exit conditions: reverse signal when opposite breakout occurs
        elif position == 1 and close[i] < lowest_low[i]:
            signals[i] = 0.0  # Exit long
            position = 0
        elif position == -1 and close[i] > highest_high[i]:
            signals[i] = 0.0  # Exit short
            position = 0
        else:
            # Hold current position or stay flat
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0