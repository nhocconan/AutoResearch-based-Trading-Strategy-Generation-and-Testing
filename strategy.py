#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_ema_volume_v2
Hypothesis: Use weekly EMA50 trend filter with Camarilla pivot levels from daily timeframe.
Enter long at S3 level or short at R3 level with volume confirmation.
Only trade in direction of weekly trend to avoid counter-trend whipsaw.
Targets 15-30 trades per year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_ema_volume_v2"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    s3 = close - range_val * 1.1 / 4
    s2 = close - range_val * 1.1 / 6
    s1 = close - range_val * 1.1 / 12
    r1 = close + range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    return s1, s2, s3, r1, r2, r3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    s1, s2, s3, r1, r2, r3 = calculate_camarilla(prev_high, prev_low, prev_close)
    
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if trend filter not ready
        if np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs weekly EMA50
        above_ema50 = close[i] > ema50_1w_aligned[i]
        below_ema50 = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S2 or trend turns bearish
            if close[i] < s2_aligned[i] or below_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R2 or trend turns bullish
            if close[i] > r2_aligned[i] or above_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long: price at S3 level with volume confirmation and bullish trend
            if (close[i] <= s3_aligned[i] * 1.002 and  # Allow small buffer
                volume_filter[i] and 
                above_ema50):
                position = 1
                signals[i] = 0.25
            # Enter short: price at R3 level with volume confirmation and bearish trend
            elif (close[i] >= r3_aligned[i] * 0.998 and   # Allow small buffer
                  volume_filter[i] and 
                  below_ema50):
                position = -1
                signals[i] = -0.25
    
    return signals