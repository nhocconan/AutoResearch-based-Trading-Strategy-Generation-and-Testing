#!/usr/bin/env python3
"""
4h_1d_Weekly_Range_Breakout
Hypothesis: Use weekly range (Monday's high/low) as key support/resistance on 4h timeframe.
Enter long when price breaks above weekly high with volume confirmation (>1.5x average),
enter short when price breaks below weekly low. Uses weekly structure to capture institutional levels.
Targets 20-40 trades per year by requiring strong breakouts at meaningful levels.
Works in both bull/bear markets as breakouts capture momentum in any regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Weekly_Range_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY INDICATORS: Weekly range (Monday's high/low) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high and low from weekly candles
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align to 4h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
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
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        long_breakout = (close[i] > weekly_high_aligned[i]) and vol_confirm
        short_breakout = (close[i] < weekly_low_aligned[i]) and vol_confirm
        
        # Exit conditions: reversal back inside weekly range
        exit_long = close[i] < weekly_low_aligned[i]
        exit_short = close[i] > weekly_high_aligned[i]
        
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