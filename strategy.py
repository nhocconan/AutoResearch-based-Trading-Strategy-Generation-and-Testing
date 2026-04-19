#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_Volume_V1
Hypothesis: 6h breakouts of weekly (Monday) pivot levels (R1/S1) with volume confirmation.
Weekly pivots provide strong institutional support/resistance that holds across market regimes.
Breakouts with volume indicate genuine institutional participation, reducing false signals.
Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
Works in bull/bear via volume confirmation and pivot-level structure.
"""

name = "6h_WeeklyPivot_Breakout_Volume_V1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data (resampled from daily for pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Group by week starting Monday
    df_1d = df_1d.copy()
    df_1d['week_start'] = df_1d.index.to_series().dt.to_period('W').dt.start_time
    
    # Get previous week's OHLC for pivot calculation
    weekly_agg = df_1d.groupby('week_start').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).shift(1)  # Previous week's data
    
    if len(weekly_agg) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels
    ph = weekly_agg['high'].values  # Previous week high
    pl = weekly_agg['low'].values   # Previous week low
    pc = weekly_agg['close'].values # Previous week close
    
    # Weekly pivot point and support/resistance levels
    pw = (ph + pl + pc) / 3  # Weekly pivot
    r1 = 2 * pw - pl         # Weekly R1
    s1 = 2 * pw - ph         # Weekly S1
    r4 = pw + 3 * (ph - pl)  # Weekly R4 (strong breakout)
    s4 = pw - 3 * (ph - pl)  # Weekly S4 (strong breakdown)
    
    # Align weekly levels to 6h timeframe
    # Need to map each 6h bar to its corresponding week's levels
    week_start_series = df_1d.index.to_series().dt.to_period('W').dt.start_time
    week_start_ffilled = week_start_series.ffill()
    
    # Create mapping from date to weekly levels
    week_to_levels = {}
    for idx, week_start in enumerate(weekly_agg.index):
        week_to_levels[week_start] = {
            'r1': r1[idx],
            's1': s1[idx],
            'r4': r4[idx],
            's4': s4[idx]
        }
    
    # Map each daily bar to its weekly levels
    r1_daily = np.full(len(df_1d), np.nan)
    s1_daily = np.full(len(df_1d), np.nan)
    r4_daily = np.full(len(df_1d), np.nan)
    s4_daily = np.full(len(df_1d), np.nan)
    
    for i, week_start in enumerate(week_start_ffilled.values):
        if week_start in week_to_levels:
            levels = week_to_levels[week_start]
            r1_daily[i] = levels['r1']
            s1_daily[i] = levels['s1']
            r4_daily[i] = levels['r4']
            s4_daily[i] = levels['s4']
    
    # Align weekly levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_daily)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_daily)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_daily)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_daily)
    
    # Volume confirmation: volume > 1.8 * 20-period average (stricter for fewer trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume confirmation
            if (close[i] > r1_6h[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume confirmation
            elif (close[i] < s1_6h[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly S1 or strong breakdown below S4
            if (close[i] < s1_6h[i]) or (close[i] < s4_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly R1 or strong breakout above R4
            if (close[i] > r1_6h[i]) or (close[i] > r4_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals