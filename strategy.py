#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume spike confirmation
# Designed for 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag.
# Donchian provides clear price structure, EMA50 defines 12h trend regime,
# volume spike filters for institutional participation. Works in bull/bear via trend filter.

name = "4h_Donchian20_12hEMA50_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume SMA (20-period) for volume spike detection
    volume_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_sma20[i]) or np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 12h EMA50 uptrend AND volume spike
            if (close[i] > highest_20[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 1.5 * volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND 12h EMA50 downtrend AND volume spike
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 1.5 * volume_sma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR 12h trend turns down
            if (close[i] <= highest_20[i] and close[i] >= lowest_20[i]) or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR 12h trend turns up
            if (close[i] <= highest_20[i] and close[i] >= lowest_20[i]) or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals