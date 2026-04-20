#!/usr/bin/env python3
# 6h_1d_WeeklyPivot_Breakout_VolumeTrend
# Hypothesis: On 6h timeframe, trade breakouts from 1d-derived weekly pivot levels with volume spike confirmation and 1d EMA trend filter.
# Uses weekly pivot points calculated from weekly high/low/close to define R1/S1 levels.
# Breakouts are confirmed by volume > 2x 20-period average and price beyond 0.5% buffer around weekly R1/S1.
# Trend filter uses 1d EMA34 to align with daily trend. Designed to work in both bull and bear markets by aligning with higher timeframe trends.
# Targets 15-30 trades per year to minimize fee drag.

name = "6h_1d_WeeklyPivot_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Group daily data into weeks (starting Monday)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high, low, close using expanding window (simplified)
    # For each day, weekly high = max of last 5 days (approximation)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=1).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=1).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=1).last().values
    
    # Weekly pivot point and range
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    
    # Weekly R1 and S1 levels (standard pivot)
    s1_weekly = 2 * weekly_pivot - weekly_high
    r1_weekly = 2 * weekly_pivot - weekly_low
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly levels and 1d EMA to 6h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_weekly)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_weekly)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R1, volume spike, and price above 1d EMA34 (uptrend)
            if (close[i] > r1_aligned[i] * 1.005 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S1, volume spike, and price below 1d EMA34 (downtrend)
            elif (close[i] < s1_aligned[i] * 0.995 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S1 or trend reversal (below EMA34)
            if close[i] < s1_aligned[i] * 0.995 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R1 or trend reversal (above EMA34)
            if close[i] > r1_aligned[i] * 1.005 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals