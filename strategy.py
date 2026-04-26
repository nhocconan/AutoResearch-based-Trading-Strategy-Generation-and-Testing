#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_VolumeConfirm
Hypothesis: On 6h timeframe, trade weekly Camarilla pivot breakouts (R4/S4) with 6h Donchian(20) confirmation and volume spike (>2x 20-median). Weekly pivot provides strong institutional levels from prior week, Donchian confirms breakout momentum, volume spike validates institutional participation. Designed for low frequency (target 50-150 trades over 4 years) to minimize fee drag. Works in bull via breakout continuation and in bear by requiring strong volume confirmation to avoid false breakouts during low volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # We'll use 1w data to get weekly OHLC, then compute Camarilla for that week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get weekly OHLC arrays (each value represents the complete week)
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    weekly_range = weekly_high - weekly_low
    r4_weekly = weekly_close + weekly_range * 1.1 / 2
    s4_weekly = weekly_close - weekly_range * 1.1 / 2
    r3_weekly = weekly_close + weekly_range * 1.1 / 4
    s3_weekly = weekly_close - weekly_range * 1.1 / 4
    r1_weekly = weekly_close + weekly_range * 1.1 / 12
    s1_weekly = weekly_close - weekly_range * 1.1 / 12
    
    # Align weekly Camarilla levels to 6h timeframe (completed weekly bars only)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_weekly)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_weekly)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_weekly)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_weekly)
    
    # 6h Donchian(20) for breakout confirmation
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2x 20-period median (robust to outliers)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period for Donchian and volume median)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above weekly R4 with Donchian breakout and volume spike
        long_condition = (close[i] > r4_aligned[i]) and (close[i] > donchian_high[i]) and volume_spike[i]
        # Short logic: break below weekly S4 with Donchian breakout and volume spike
        short_condition = (close[i] < s4_aligned[i]) and (close[i] < donchian_low[i]) and volume_spike[i]
        
        # Exit logic: return to weekly R1/S1 levels (mean reversion to weekly value area)
        exit_long = close[i] < r1_aligned[i]
        exit_short = close[i] > s1_aligned[i]
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0