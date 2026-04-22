#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# - Donchian breakout captures momentum in both bull and bear markets
# - Weekly pivot (from 1d data) provides directional bias: long above weekly pivot, short below
# - Volume > 1.5x 20-period average confirms breakout strength
# - Works in trending and ranging markets by combining price structure with institutional levels
# - Discrete position sizing (0.25) minimizes fee churn

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for weekly pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from daily data (using prior week's data)
    # Weekly high = max of last 5 daily highs, weekly low = min of last 5 daily lows
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly resistance 1 = (2 * weekly_pivot) - weekly_low
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    # Weekly support 1 = (2 * weekly_pivot) - weekly_high
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high AND above weekly pivot AND volume spike
            long_breakout = close[i] > donchian_high[i]
            above_pivot = close[i] > weekly_pivot_aligned[i]
            volume_spike = volume[i] > 1.5 * vol_avg_20[i]
            
            if long_breakout and above_pivot and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND below weekly pivot AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price breaks opposite Donchian level or crosses weekly pivot
            if position == 1:
                # Exit long: Price breaks below Donchian low OR crosses below weekly pivot
                if close[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price breaks above Donchian high OR crosses above weekly pivot
                if close[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0