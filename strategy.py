#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
# Donchian breakouts capture trend continuation. Weekly EMA50 ensures alignment with higher timeframe trend.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Designed for 1d timeframe to capture multi-day trends in both bull and bear markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50-period EMA on weekly close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above upper Donchian band + weekly uptrend + volume
        if close[i] > high_max[i] and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short entry: price breaks below lower Donchian band + weekly downtrend + volume
        elif close[i] < low_min[i] and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or loss of trend/volume
        elif position == 1 and (close[i] < low_min[i] or close[i] < ema50_1w_aligned[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > high_max[i] or close[i] > ema50_1w_aligned[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0