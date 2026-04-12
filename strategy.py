#/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_v3
Hypothesis: On 4h timeframe, enter long when price breaks above daily Camarilla R4 with volume confirmation (>1.5x average), enter short when price breaks below daily Camarilla S4. Uses daily timeframe for structure and volume filter to avoid false breakouts. Targets 20-50 trades per year by requiring strong breakouts at extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_v3"
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
    
    # === DAILY INDICATORS: Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R4 and S4 levels (extreme reversal levels)
    r4 = close_1d + range_1d * 1.1 / 2
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        long_breakout = (close[i] > r4_aligned[i]) and vol_confirm
        short_breakout = (close[i] < s4_aligned[i]) and vol_confirm
        
        # Exit conditions: reversal back inside Camarilla R3-S3 range
        r3 = close_1d + range_1d * 1.1 / 4
        s3 = close_1d - range_1d * 1.1 / 4
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        
        exit_long = close[i] < r3_aligned[i]
        exit_short = close[i] > s3_aligned[i]
        
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