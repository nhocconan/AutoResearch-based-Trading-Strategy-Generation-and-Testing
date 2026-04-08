# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_1d_camarilla_pivot_volume_v3
Hypothesis: 4h price breaking above/below 12h Camarilla pivot levels (R1/S1) with volume confirmation
and intramarket session filter creates high-probability breakout trades in both bull and bear markets.
Uses 1d timeframe for Camarilla calculation (proper support/resistance levels) and 4h for entry timing.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v3"
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
    open_prices = prices['open'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculations
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    s2 = close_1d - (range_1d * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: volume > 1.3x average of last 20 periods (5 sessions)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    # Session filter: active London/NY overlap (12:00-16:00 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 12) & (hours <= 16)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S1 or loses upward momentum
            if close[i] < s1_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R1 or loses downward momentum
            if close[i] > r1_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above R1 with volume and session filter
            if (close[i] > r1_1d_aligned[i] and 
                open_prices[i] <= r1_1d_aligned[i] and  # Ensure breakout happened this bar
                vol_confirm[i] and session_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S1 with volume and session filter
            elif (close[i] < s1_1d_aligned[i] and 
                  open_prices[i] >= s1_1d_aligned[i] and  # Ensure breakdown happened this bar
                  vol_confirm[i] and session_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals