#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyTrend_VolumeBreakout
# Hypothesis: Use weekly pivot points as structural support/resistance, filtered by daily EMA trend and volume spikes.
# In bull markets: buy pullbacks to weekly S1/S2 in uptrend. In bear markets: sell rallies to weekly R1/R2 in downtrend.
# Weekly pivots provide multi-day structure; daily trend filters whipsaw; volume confirms institutional interest.
# Targets 15-30 trades/year on 6h timeframe to minimize fee drag.

name = "6h_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    daily_close = df_daily['close'].values
    ema_50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average (24-period for 6h: ~6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align all timeframes to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need EMA50 (50) + volume MA (24)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_daily_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from daily EMA50
        uptrend = close[i] > ema_50_daily_aligned[i]
        downtrend = close[i] < ema_50_daily_aligned[i]
        
        # Volume confirmation (2x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: pullback to weekly support in uptrend with volume
            if (close[i] <= s1_aligned[i] * 1.005 and  # within 0.5% of S1
                close[i] >= s2_aligned[i] * 0.995 and  # within 0.5% of S2
                uptrend and
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: rally to weekly resistance in downtrend with volume
            elif (close[i] >= r1_aligned[i] * 0.995 and  # within 0.5% of R1
                  close[i] <= r2_aligned[i] * 1.005 and  # within 0.5% of R2
                  downtrend and
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 4 bars (1 day)
            if bars_since_entry < 4:
                signals[i] = signals[i-1]
                continue
            
            if position == 1:
                # Long exit: break below S2 or trend change
                if close[i] < s2_aligned[i] * 0.995 or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: break above R2 or trend change
                if close[i] > r2_aligned[i] * 1.005 or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals