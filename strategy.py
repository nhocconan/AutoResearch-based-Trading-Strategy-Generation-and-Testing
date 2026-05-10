#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyTrend_VolumeBreakout
# Hypothesis: Weekly pivot points (R2/S2) act as key support/resistance; breakouts beyond these levels with daily trend alignment and volume spikes capture strong momentum. Works in bull (breakouts) and bear (mean reversion at extremes) by filtering with daily trend to avoid counter-trend trades. Weekly timeframe reduces noise, daily trend ensures alignment with intermediate momentum, and volume confirms institutional interest. Designed for low trade frequency (~15-30/year) to minimize fee drag on 6h timeframe.

name = "6h_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    range_weekly = high_weekly - low_weekly
    R2 = pivot_weekly + range_weekly
    S2 = pivot_weekly - range_weekly
    
    # Daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_daily_up = close_daily > ema34_daily
    trend_daily_down = close_daily < ema34_daily
    
    # Align weekly pivot levels to 6h
    R2_aligned = align_htf_to_ltf(prices, df_weekly, R2)
    S2_aligned = align_htf_to_ltf(prices, df_weekly, S2)
    
    # Align daily trend to 6h
    trend_daily_up_aligned = align_htf_to_ltf(prices, df_daily, trend_daily_up.astype(float))
    trend_daily_down_aligned = align_htf_to_ltf(prices, df_daily, trend_daily_down.astype(float))
    
    # Volume spike: current > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(trend_daily_up_aligned[i]) or np.isnan(trend_daily_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above R2 with daily uptrend and volume spike
            if (close[i] > R2_aligned[i] and 
                trend_daily_up_aligned[i] > 0.5 and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with daily downtrend and volume spike
            elif (close[i] < S2_aligned[i] and 
                  trend_daily_down_aligned[i] > 0.5 and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below S2 or trend fails
            if (close[i] < S2_aligned[i] or 
                trend_daily_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above R2 or trend fails
            if (close[i] > R2_aligned[i] or 
                trend_daily_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals