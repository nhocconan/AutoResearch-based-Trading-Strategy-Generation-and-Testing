#!/usr/bin/env python3
"""
6H_WeeklyPivot_Breakout_DailyTrend
Hypothesis: 6h price breaks above/below weekly pivot levels with daily EMA50 trend confirmation and volume spike.
Weekly pivots provide strong support/resistance levels that work across market regimes.
Daily EMA50 ensures alignment with intermediate trend, volume confirms breakout strength.
Targets 12-30 trades/year to minimize fee drag on 6h timeframe.
Works in bull/bear markets: breakouts capture strong moves while avoiding minor retracements.
"""
name = "6H_WeeklyPivot_Breakout_DailyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Get daily data for EMA50 and volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    r1_w = pivot_w + (range_w * 1.0)  # R1 = pivot + range
    s1_w = pivot_w - (range_w * 1.0)  # S1 = pivot - range
    
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    
    # Calculate daily EMA50 for trend direction
    close_d = df_daily['close'].values
    ema_50 = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_daily, ema_50)
    
    # Daily volume filter: current 6h volume > 1.8 x 24-period average volume
    # (24 periods of 6h = 6 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_avg * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 24)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 72 bars between trades (18 days on 6h TF) to reduce frequency
            if bars_since_exit < 72:
                continue
                
            # Long: price breaks above weekly R1 with daily EMA50 uptrend and volume spike
            if (close[i] > r1_w_aligned[i] and close[i-1] <= r1_w_aligned[i-1] and 
                close[i] > ema_50_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below weekly S1 with daily EMA50 downtrend and volume spike
            elif (close[i] < s1_w_aligned[i] and close[i-1] >= s1_w_aligned[i-1] and 
                  close[i] < ema_50_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA50 side (trend reversal)
            if position == 1 and close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals