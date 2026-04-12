#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_camarilla_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from weekly high/low/close
    # Using previous week's data (already available in df_1w)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r4 = close_1w + range_1w * 1.500
    r3 = close_1w + range_1w * 1.250
    r2 = close_1w + range_1w * 1.166
    r1 = close_1w + range_1w * 1.083
    s1 = close_1w - range_1w * 1.083
    s2 = close_1w - range_1w * 1.166
    s3 = close_1w - range_1w * 1.250
    s4 = close_1w - range_1w * 1.500
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume filter: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # warmup for volume MA
        # Skip if not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_long = close[i] > r4_aligned[i]  # Break above R4
        breakout_short = close[i] < s4_aligned[i]  # Break below S4
        
        # Fade conditions (mean reversion at extreme levels)
        fade_long = close[i] < s3_aligned[i] and close[i] > s4_aligned[i]  # Between S3 and S4
        fade_short = close[i] > r3_aligned[i] and close[i] < r4_aligned[i]  # Between R3 and R4
        
        # Volume confirmation
        vol_ok = volume_ok[i]
        
        # Exit conditions: return to pivot or opposite extreme
        exit_long = close[i] < pivot[i] or close[i] > r3_aligned[i]
        exit_short = close[i] > pivot[i] or close[i] < s3_aligned[i]
        
        # Execute trades
        if breakout_long and vol_ok and position != 1:
            position = 1
            signals[i] = 0.25
        elif fade_long and vol_ok and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and vol_ok and position != -1:
            position = -1
            signals[i] = -0.25
        elif fade_short and vol_ok and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals