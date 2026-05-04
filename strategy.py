#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d pivot direction filter and volume confirmation
# Donchian(20) breakouts capture medium-term trends on 6h timeframe.
# 1d pivot direction (based on previous day's close vs pivot) filters for institutional bias.
# Volume spike (>1.8 x 20-period EMA) ensures participation and reduces false breakouts.
# Designed for 6h timeframe targeting 75-150 total trades over 4 years (19-37/year).
# Works in bull markets via trend-aligned breakouts and in bear markets via filtered mean-reversion at extremes.

name = "6h_Donchian20_1dPivotDirection_VolumeSpike"
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
    
    # Get 1d data for pivot direction filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d pivot point and direction from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's pivot point: PP = (H + L + C) / 3
    pp_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    # Shift to align with current day: yesterday's PP affects today's bias
    pp_1d_shifted = np.empty_like(pp_1d)
    pp_1d_shifted[0] = np.nan  # First day has no yesterday
    pp_1d_shifted[1:] = pp_1d[:-1]
    
    # Pivot direction: 1 if close > PP (bullish bias), -1 if close < PP (bearish bias)
    pivot_direction_1d = np.where(close_1d > pp_1d_shifted, 1, -1)
    pivot_direction_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_direction_1d)
    
    # Donchian(20) channels on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_direction_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation: current volume > 1.8 x 20-period EMA
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0:
            # Long: Close breaks above Donchian upper + bullish 1d pivot direction + volume spike
            if (close[i] > highest_high[i] and 
                pivot_direction_1d_aligned[i] == 1 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower + bearish 1d pivot direction + volume spike
            elif (close[i] < lowest_low[i] and 
                  pivot_direction_1d_aligned[i] == -1 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close drops below Donchian midpoint OR pivot direction turns bearish
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] < midpoint or pivot_direction_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close rises above Donchian midpoint OR pivot direction turns bullish
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] > midpoint or pivot_direction_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals