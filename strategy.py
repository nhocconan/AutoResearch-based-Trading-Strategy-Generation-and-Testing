# 6h_1w_cam_pivot_reversion_v1
# Fade at weekly pivot extremes (S3/R3) with mean reversion, continue at S4/R4 breakouts.
# Uses weekly Camarilla pivot levels as dynamic support/resistance.
# Works in both bull and bear markets: mean reversion in ranges, breakout continuation in trends.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Pivot point and ranges
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    
    # Camarilla levels
    r4_w = close_w + range_w * 1.1 / 2
    r3_w = close_w + range_w * 1.1 / 4
    r2_w = close_w + range_w * 1.1 / 6
    r1_w = close_w + range_w * 1.1 / 12
    s1_w = close_w - range_w * 1.1 / 12
    s2_w = close_w - range_w * 1.1 / 6
    s3_w = close_w - range_w * 1.1 / 4
    s4_w = close_w - range_w * 1.1 / 2
    
    # Align weekly levels to 6h timeframe
    r4_w_aligned = align_htf_to_ltf(prices, df_1w, r4_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_1w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_1w, s3_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_ok = prices['volume'].values > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_w_aligned[i]) or np.isnan(r3_w_aligned[i]) or
            np.isnan(s3_w_aligned[i]) or np.isnan(s4_w_aligned[i]) or
            np.isnan(vol_ok[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion at S3/R3 with volume
        long_reversion = (close[i] <= s3_w_aligned[i]) and vol_ok[i]
        short_reversion = (close[i] >= r3_w_aligned[i]) and vol_ok[i]
        
        # Breakout continuation at S4/R4 with volume
        long_breakout = (close[i] >= s4_w_aligned[i]) and vol_ok[i]
        short_breakout = (close[i] <= r4_w_aligned[i]) and vol_ok[i]
        
        # Exit conditions: opposite S3/R3 level or volume fails
        exit_long = position == 1 and (
            (close[i] >= r3_w_aligned[i]) or 
            (not vol_ok[i])
        )
        exit_short = position == -1 and (
            (close[i] <= s3_w_aligned[i]) or 
            (not vol_ok[i])
        )
        
        # Enter long on mean reversion or breakout
        if (long_reversion or long_breakout) and position != 1:
            position = 1
            signals[i] = position_size
        # Enter short on mean reversion or breakout
        elif (short_reversion or short_breakout) and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_cam_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0