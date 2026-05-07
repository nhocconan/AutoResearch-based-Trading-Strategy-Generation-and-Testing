#!/usr/bin/env python3
"""
6H_WeeklyPivot_DailyTrend_Volume
Hypothesis: 6h price breaks above/below weekly pivot with daily EMA50 trend confirmation and volume spike.
Weekly pivot acts as institutional support/resistance. EMA50 filters trend direction.
Volume spike validates breakout strength. Works in bull/bear markets by capturing strong moves
while avoiding minor retracements. Targets 15-35 trades/year to minimize fee drag on 6h timeframe.
"""
name = "6H_WeeklyPivot_DailyTrend_Volume"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (classic: (H+L+C)/3)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # Get daily data for EMA50 trend and volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend direction
    close_daily_series = pd.Series(df_daily['close'])
    ema_50 = close_daily_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_daily, ema_50)
    
    # Volume filter: current 6h volume > 1.8 x 24-period average volume
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_avg * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 24)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 72 bars between trades (18 days on 6h TF) to reduce frequency
            if bars_since_exit < 72:
                continue
                
            # Long: price breaks above weekly pivot with daily EMA50 uptrend and volume spike
            if (close[i] > weekly_pivot_aligned[i] and close[i-1] <= weekly_pivot_aligned[i-1] and 
                close[i] > ema_50_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below weekly pivot with daily EMA50 downtrend and volume spike
            elif (close[i] < weekly_pivot_aligned[i] and close[i-1] >= weekly_pivot_aligned[i-1] and 
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