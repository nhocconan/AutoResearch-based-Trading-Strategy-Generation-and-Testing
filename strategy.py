#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_WeeklyPivotFilter
Hypothesis: On 6h timeframe, enter breakout trades at Camarilla R3/S3 levels with 1d EMA34 trend filter and weekly pivot direction filter. 
Weekly pivot acts as regime filter: only take longs when price above weekly pivot, shorts when below. 
This reduces false breakouts in choppy markets and aligns with higher timeframe structure. 
Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.0, ±0.25) to minimize fee drag.
Works in bull/bear markets by combining 1d trend with weekly structural bias.
"""

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla R3/S3 from prior daily bar
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_open = np.roll(df_1d['open'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    prior_open[0] = np.nan
    
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_hl = prior_high - prior_low
    r3 = pivot + range_hl * 1.1 / 4   # R3 level
    s3 = pivot - range_hl * 1.1 / 4   # S3 level
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load weekly data for pivot filter
    df_1w = get_htf_data(prices, '1w')
    prior_week_high = np.roll(df_1w['high'].values, 1)
    prior_week_low = np.roll(df_1w['low'].values, 1)
    prior_week_close = np.roll(df_1w['close'].values, 1)
    prior_week_high[0] = np.nan
    prior_week_low[0] = np.nan
    prior_week_close[0] = np.nan
    
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 1-day shift + 34-day EMA + 1-week shift)
    start_idx = max(1 + 34, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 + bullish daily trend + above weekly pivot + volume spike
        if (close[i] > r3_aligned[i] and 
            close[i] > ema_34_1d_aligned[i] and 
            close[i] > weekly_pivot_aligned[i] and 
            volume_spike[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + bearish daily trend + below weekly pivot + volume spike
        elif (close[i] < s3_aligned[i] and 
              close[i] < ema_34_1d_aligned[i] and 
              close[i] < weekly_pivot_aligned[i] and 
              volume_spike[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level (R1/S1) for quicker exit
        elif position == 1 and close[i] < (pivot - range_hl * 1.1 / 12)[i]:  # S1 level
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (pivot + range_hl * 1.1 / 12)[i]:  # R1 level
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

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_WeeklyPivotFilter"
timeframe = "6h"
leverage = 1.0