#!/usr/bin/env python3
"""
12h_pivots_1w_trend_v1
Hypothesis: On 12-hour timeframe, use weekly pivot points with trend filter from weekly EMA. 
Enter long when price is above weekly pivot AND above weekly EMA(50) with volume confirmation. 
Enter short when price is below weekly pivot AND below weekly EMA(50) with volume confirmation. 
Exit when price crosses back across weekly pivot. Designed for low frequency (15-30 trades/year) 
to capture major trend moves while minimizing whipsaw in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_pivots_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Standard pivot point calculation
    pivot = (w_high + w_low + w_close) / 3
    r1 = 2 * pivot - w_low
    s1 = 2 * pivot - w_high
    r2 = pivot + (w_high - w_low)
    s2 = pivot - (w_high - w_low)
    r3 = w_high + 2 * (pivot - w_low)
    s3 = w_low - 2 * (w_high - pivot)
    
    # Weekly EMA(50) for trend filter
    w_close_series = pd.Series(w_close)
    w_ema50 = w_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, w_ema50)
    
    # Calculate 30-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA and volume average warmup
        # Skip if weekly data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 30-period average
        vol_confirm = volume[i] > 1.3 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price crosses below weekly pivot
            if close[i] < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price crosses above weekly pivot
            if close[i] > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above pivot AND above EMA50 with volume confirmation
            long_entry = (close[i] > pivot_aligned[i]) and (close[i] > ema50_aligned[i]) and vol_confirm
            # Short entry: price below pivot AND below EMA50 with volume confirmation
            short_entry = (close[i] < pivot_aligned[i]) and (close[i] < ema50_aligned[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals