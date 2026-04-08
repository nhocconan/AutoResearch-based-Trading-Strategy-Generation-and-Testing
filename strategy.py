#!/usr/bin/env python3
"""
12h_1d_1w_camarilla_volume_v2
Hypothesis: Camarilla pivot levels from daily timeframe with volume confirmation and weekly trend filter.
- Entry: Price touches Camarilla S3 (long) or R3 (short) with volume spike
- Trend filter: Weekly EMA(50) direction - only trade long in weekly uptrend, short in downtrend
- Exit: Price reaches Camarilla C level (midpoint) or opposite S/R level
- Position sizing: 0.25
- Designed to work in ranging markets (camarilla reversals) and trending markets (weekly filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_volume_v2"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    c = (high + low + close) / 3
    s3 = c - (range_val * 1.1 / 4)
    s2 = c - (range_val * 1.1 / 6)
    s1 = c - (range_val * 1.1 / 12)
    r1 = c + (range_val * 1.1 / 12)
    r2 = c + (range_val * 1.1 / 6)
    r3 = c + (range_val * 1.1 / 4)
    return s1, s2, s3, r1, r2, r3, c

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    s1_1d = np.full_like(close_1d, np.nan)
    s2_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    r2_1d = np.full_like(close_1d, np.nan)
    r3_1d = np.full_like(close_1d, np.nan)
    c_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        s1, s2, s3, r1, r2, r3, c = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        r1_1d[i] = r1
        r2_1d[i] = r2
        r3_1d[i] = r3
        c_1d[i] = c
    
    # Align Camarilla levels to 12h
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema_50_1w
    trend_1w_down = close_1w < ema_50_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align weekly trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(c_1d_aligned[i]) or np.isnan(trend_1w_up_aligned[i]) or
            np.isnan(trend_1w_down_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reaches Camarilla C level OR R3 level (take profit)
            if close[i] >= c_1d_aligned[i] or close[i] >= r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price reaches Camarilla C level OR S3 level (take profit)
            if close[i] <= c_1d_aligned[i] or close[i] <= s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price touches S3 level + weekly uptrend + volume spike
            if (abs(close[i] - s3_1d_aligned[i]) < 0.001 * close[i]) and trend_1w_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches R3 level + weekly downtrend + volume spike
            elif (abs(close[i] - r3_1d_aligned[i]) < 0.001 * close[i]) and trend_1w_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals