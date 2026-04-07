#!/usr/bin/env python3
"""
6h_weekly_pivot_12h_trend_volume_v1
Hypothesis: On 6-hour timeframe, use weekly pivot points (calculated from previous week) with 12h EMA trend filter and volume confirmation. Enter long when price breaks above weekly R1 in uptrend with volume > 1.3x average, short when price breaks below weekly S1 in downtrend with volume > 1.3x average. Exit when price touches opposite weekly pivot level (S1 for long, R1 for short). Weekly pivots provide institutional support/resistance that adapt to changing market conditions across bull/bear cycles. Designed for low frequency (12-37 trades/year) to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    w_pivot = (w_high + w_low + w_close) / 3
    w_range = w_high - w_low
    
    # Weekly support/resistance levels
    r1 = 2 * w_pivot - w_low
    s1 = 2 * w_pivot - w_high
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get EMA50 from 12h for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if weekly data not available
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs 12h EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below S1
            if close[i] <= s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above R1
            if close[i] >= r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R1 in uptrend with volume confirmation
            long_entry = (close[i] > r1_aligned[i]) and uptrend and vol_confirm
            # Short entry: price breaks below S1 in downtrend with volume confirmation
            short_entry = (close[i] < s1_aligned[i]) and downtrend and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals