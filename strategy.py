#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_DailyVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous weekly bar
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    weekly_range = high_weekly - low_weekly
    
    # Weekly resistance and support levels (R1, S1)
    weekly_r1 = 2 * weekly_pivot - low_weekly
    weekly_s1 = 2 * weekly_pivot - high_weekly
    
    # Align weekly levels to 6h timeframe (using previous weekly bar's values)
    r1_6h = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_6h = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema_50_daily = pd.Series(close_daily).ewm(span=50, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume filter: current volume > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 AND above daily EMA50 (uptrend) AND volume spike
            if close[i] > r1_6h[i] and close[i] > ema_50_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 AND below daily EMA50 (downtrend) AND volume spike
            elif close[i] < s1_6h[i] and close[i] < ema_50_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly S1 OR below daily EMA50 (trend change)
            if close[i] < s1_6h[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly R1 OR above daily EMA50 (trend change)
            if close[i] > r1_6h[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals