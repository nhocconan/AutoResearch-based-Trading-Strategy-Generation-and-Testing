#!/usr/bin/env python3
"""
4h_12h_pivot_volume_v1
Hypothesis: Use 12h pivot levels for key support/resistance and 4h price action with volume confirmation for breakout trading.
Long when 4h price breaks above 12h R1 with volume confirmation and 12h trend up.
Short when 4h price breaks below 12h S1 with volume confirmation and 12h trend down.
Designed to work in both bull (breakouts) and bear (reversals at key levels) markets.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot calculation and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h pivot points (standard floor trader pivots)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h pivot point
    pivot_point = (high_12h + low_12h + close_12h) / 3.0
    # 12h resistance and support levels
    r1 = 2 * pivot_point - low_12h
    s1 = 2 * pivot_point - high_12h
    r2 = pivot_point + (high_12h - low_12h)
    s2 = pivot_point - (high_12h - low_12h)
    
    # Align 12h pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # 12h trend using EMA (21-period)
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 12h S1 or trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 12h R1 or trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 12h R1 with volume and uptrend
            if close[i] > r1_aligned[i] and vol_confirm[i] and close[i] > ema_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 12h S1 with volume and downtrend
            elif close[i] < s1_aligned[i] and vol_confirm[i] and close[i] < ema_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals