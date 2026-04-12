#/usr/bin/env python3
"""
1d_1w_Aziz_Breakout_v1
Hypothesis: On daily timeframe, enter long when price breaks above weekly resistance with volume confirmation, enter short when breaks below weekly support. Uses weekly timeframe for structure and volume filter to avoid false breakouts. Targets 10-25 trades per year by requiring strong breakouts at weekly extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Aziz_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY INDICATORS: Support/Resistance levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly range and key levels
    range_1w = high_1w - low_1w
    # Weekly resistance: previous week high + 0.382 * range (Fibonacci)
    resistance = high_1w + 0.382 * range_1w
    # Weekly support: previous week low - 0.382 * range (Fibonacci)
    support = low_1w - 0.382 * range_1w
    
    # Align to daily timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_1w, resistance)
    support_aligned = align_htf_to_ltf(prices, df_1w, support)
    
    # Volume average (20-period) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        long_breakout = (close[i] > resistance_aligned[i]) and vol_confirm
        short_breakout = (close[i] < support_aligned[i]) and vol_confirm
        
        # Exit conditions: reversal back to midpoint of weekly range
        midpoint = (high_1w + low_1w) / 2
        midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint)
        
        exit_long = close[i] < midpoint_aligned[i]
        exit_short = close[i] > midpoint_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals