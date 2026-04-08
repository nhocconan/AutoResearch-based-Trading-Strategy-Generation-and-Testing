#!/usr/bin/env python3
"""
12h_1w_1d_camarilla_volume_v3
Hypothesis: Camarilla pivot levels from 1w + volume confirmation on 12h + 1d trend filter.
- Entry: Price touches Camarilla S3 (support) or R3 (resistance) from 1w pivot + volume > 1.5x 20-period average
- Trend filter: 1d EMA(50) direction (only long when 1d close > EMA50, only short when 1d close < EMA50)
- Exit: Price crosses back through pivot point or trend reverses
- Position sizing: 0.25 long, -0.25 short
- Designed for range-bound markets (mean reversion at extremes) with trend filter to avoid fighting strong trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_volume_v3"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    pivot = (high + low + close) / 3
    r4 = pivot + (range_val * 1.1 / 2)
    r3 = pivot + (range_val * 1.1 / 4)
    r2 = pivot + (range_val * 1.1 / 6)
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4,
        's1': s1, 's2': s2, 's3': s3, 's4': s4
    }

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_levels = []
    for i in range(len(close_1w)):
        levels = calculate_camarilla(high_1w[i], low_1w[i], close_1w[i])
        camarilla_levels.append(levels)
    
    # Extract S3 and R3 levels
    s3_levels = np.array([levels['s3'] for levels in camarilla_levels])
    r3_levels = np.array([levels['r3'] for levels in camarilla_levels])
    pivot_levels = np.array([levels['pivot'] for levels in camarilla_levels])
    
    # Align 1w Camarilla levels to 12h
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_levels)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_levels)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema_50_1d
    trend_1d_down = close_1d < ema_50_1d
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # Volume filter on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses back through pivot OR 1d trend turns down
            if (close[i] > pivot_aligned[i]) or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price crosses back through pivot OR 1d trend turns up
            if (close[i] < pivot_aligned[i]) or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price touches S3 from below + 1d uptrend + volume
            if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) and trend_1d_up_aligned[i] and volume_filter[i]:
                # Confirm touch and bounce
                if i > start_idx and low[i-1] <= s3_aligned[i-1] and close[i-1] <= s3_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
            # Short entry: Price touches R3 from above + 1d downtrend + volume
            elif (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) and trend_1d_down_aligned[i] and volume_filter[i]:
                # Confirm touch and rejection
                if i > start_idx and high[i-1] >= r3_aligned[i-1] and close[i-1] >= r3_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals