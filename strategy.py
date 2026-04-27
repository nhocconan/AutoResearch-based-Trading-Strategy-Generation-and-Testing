#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly MA50 trend filter and volume confirmation.
# Uses 1d Donchian channels for breakout signals and 1w MA50 for trend filter to avoid counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Works in bull markets (breakouts continue) and bear markets (trend filter prevents false breakouts).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50-period MA on weekly close for trend filter
    ma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    ma50_1w_aligned = align_htf_to_ltf(prices, df_1w, ma50_1w)
    
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
            np.isnan(ma50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price breaks above upper Donchian band in uptrend with volume
        if close[i] > high_roll[i] and close[i] > ma50_1w_aligned[i] and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short signal: price breaks below lower Donchian band in downtrend with volume
        elif close[i] < low_roll[i] and close[i] < ma50_1w_aligned[i] and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or loss of trend/volume
        elif position == 1 and (close[i] < low_roll[i] or close[i] < ma50_1w_aligned[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > high_roll[i] or close[i] > ma50_1w_aligned[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        # Hold current position
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0