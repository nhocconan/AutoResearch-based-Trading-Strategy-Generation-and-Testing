# Solution
#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_DoubleConfirm"
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
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly OHLC from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly range for pivot calculation
    range_1w = high_1w - low_1w
    
    # Weekly pivot point and key levels (R3, S3)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r3 = weekly_pivot + range_1w * 1.1
    weekly_s3 = weekly_pivot - range_1w * 1.1
    
    # Align weekly levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r3_6h = align_htf_to_ltf(prices, df_1w, weekly_r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1w)  # Use weekly close for EMA (more stable)
    ema_1w = close_1d_series.ewm(span=50, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot AND above weekly EMA50 AND volume filter
            if close[i] > pivot_6h[i] and close[i] > ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND below weekly EMA50 AND volume filter
            elif close[i] < pivot_6h[i] and close[i] < ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly pivot OR below weekly EMA50
            if close[i] < pivot_6h[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly pivot OR above weekly EMA50
            if close[i] > pivot_6h[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals