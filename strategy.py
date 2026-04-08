#!/usr/bin/env python3
"""
6h_camarilla_weekly_pivot_v1
Hypothesis: Combines daily Camarilla pivot levels with weekly pivot direction for 6h timeframe.
- Long when price breaks above R3 with weekly pivot bullish and volume confirmation
- Short when price breaks below S3 with weekly pivot bearish and volume confirmation
- Uses Camarilla levels from daily timeframe and weekly pivot for trend filter
- Targets 15-30 trades/year to avoid fee drag while capturing meaningful breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 1.5x 24-period average (4 days of 6h bars)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    R4 = close_1d + 1.1 * (high_1d - low_1d) * 1.5 / 2
    S4 = close_1d - 1.1 * (high_1d - low_1d) * 1.5 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get weekly data for pivot direction (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot point: (H+L+C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Weekly trend: bullish if close > pivot, bearish if close < pivot
    weekly_bullish = close_1w > weekly_pivot
    weekly_bearish = close_1w < weekly_pivot
    
    # Align weekly trend to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price drops below R3 or weekly turns bearish
            if close[i] < r3_aligned[i] or weekly_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price rises above S3 or weekly turns bullish
            if close[i] > s3_aligned[i] or weekly_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above R3, weekly bullish, volume surge
            if (close[i] > r3_aligned[i] and 
                weekly_bullish_aligned[i] > 0.5 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S3, weekly bearish, volume surge
            elif (close[i] < s3_aligned[i] and 
                  weekly_bearish_aligned[i] > 0.5 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals