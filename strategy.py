#!/usr/bin/env python3
# 4h_12h_pivot_volume_v1
# Hypothesis: 4h price breaking above/below 12h pivot point resistance/support levels
# (R1/S1) with volume confirmation creates high-probability breakout trades.
# Uses 12h timeframe for pivot calculation (proper support/resistance levels) and
# 4h for entry timing. Works in both bull/bear markets by trading breakouts in
# direction of prevailing trend. Target: 20-40 trades/year.

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
    open_prices = prices['open'].values
    
    # Get 12h data for pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_12h + low_12h + close_12h) / 3.0
    # Resistance 1 = (2 * P) - L
    r1 = (2 * pivot) - low_12h
    # Support 1 = (2 * P) - H
    s1 = (2 * pivot) - high_12h
    # Resistance 2 = P + (H - L)
    r2 = pivot + (high_12h - low_12h)
    # Support 2 = P - (H - L)
    s2 = pivot - (high_12h - low_12h)
    
    # Align pivot levels to 4h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # Volume confirmation: volume > 1.5x average of last 12 periods (3 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pivot_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or \
           np.isnan(s1_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S1 or loses upward momentum
            if close[i] < s1_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R1 or loses downward momentum
            if close[i] > r1_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above R1 with volume
            if (close[i] > r1_12h_aligned[i] and 
                open_prices[i] <= r1_12h_aligned[i] and  # Ensure breakout happened this bar
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S1 with volume
            elif (close[i] < s1_12h_aligned[i] and 
                  open_prices[i] >= s1_12h_aligned[i] and  # Ensure breakdown happened this bar
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals