#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Uses Donchian channels for breakout signals, 12h EMA50 for trend alignment, and volume > 1.5x 20-period average for confirmation.
# Designed to capture strong trends while avoiding choppy markets. Target: 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets by following the 12h trend direction only.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high and low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
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
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian upper band + uptrend + volume
        if close[i] > high_roll[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short entry: price breaks below Donchian lower band + downtrend + volume
        elif close[i] < low_roll[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or loss of trend/volume
        elif position == 1 and (close[i] < ema50_12h_aligned[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > ema50_12h_aligned[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0