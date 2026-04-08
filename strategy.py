#!/usr/bin/env python3
# 6h_camarilla_pivot_1d_trend_volume_v1
# Hypothesis: Uses Camarilla pivot levels from 1d with 12h trend filter and volume confirmation.
# Goes long at S3 bounce in uptrend (price > 12h EMA50) with volume surge.
# Goes short at R3 rejection in downtrend (price < 12h EMA50) with volume surge.
# Designed for low trade frequency (12-37/year) to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    r3_1d = pivot_1d + (range_1d * 1.1)
    s3_1d = pivot_1d - (range_1d * 1.1)
    r4_1d = pivot_1d + (range_1d * 1.5)
    s4_1d = pivot_1d - (range_1d * 1.5)
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h trend filter: EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_12h_aligned[i]
        daily_downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 or trend changes
            if close[i] < s3_1d_aligned[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or trend changes
            if close[i] > r3_1d_aligned[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: bounce at S3 in uptrend
                if daily_uptrend and close[i] > s3_1d_aligned[i] and close[i-1] <= s3_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: rejection at R3 in downtrend
                elif daily_downtrend and close[i] < r3_1d_aligned[i] and close[i-1] >= r3_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals