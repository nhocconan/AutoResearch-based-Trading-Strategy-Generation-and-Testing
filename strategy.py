#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot (based on previous week's OHLC)
    # Resample daily to weekly OHLC (using actual weekly aggregation)
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1).values  # Previous week high
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1).values    # Previous week low
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1).values  # Previous week close
    
    # Weekly pivot point and key levels
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r4 = weekly_pivot + (weekly_high - weekly_low) * 1.1  # Weekly R4 (breakout level)
    weekly_s4 = weekly_pivot - (weekly_high - weekly_low) * 1.1  # Weekly S4 (breakdown level)
    
    # Align weekly levels to 6h timeframe
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # 6h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter (6h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Enough for Donchian and weekly data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]  # Volume spike filter
        
        if position == 0:
            # Long: Price breaks above 6h Donchian high AND weekly R4 with volume
            if close[i] > donchian_high[i] and close[i] > weekly_r4_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 6h Donchian low AND weekly S4 with volume
            elif close[i] < donchian_low[i] and close[i] < weekly_s4_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below 6h Donchian low or weekly S4
            if close[i] < donchian_low[i] or close[i] < weekly_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above 6h Donchian high or weekly R4
            if close[i] > donchian_high[i] or close[i] > weekly_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals