#!/usr/bin/env python3
# 6h_12h_pivot_reversal_v1
# Strategy: 6h price rejection at 12h Camarilla pivot levels with volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: In ranging markets, price often reverses at Camarilla S3/R3 levels.
# In trending markets, breaks of S4/R4 with volume indicate continuation.
# Works in bull/bear by adapting to regime: fade at S3/R3 in range, breakout at S4/R4 in trend.
# Uses 12h Camarilla pivots calculated from prior 12h bar's OHLC.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels for each 12h bar
    # Based on prior 12h bar's OHLC (standard Camarilla calculation)
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Pivot point and ranges
    pp_12h = (h_12h + l_12h + c_12h) / 3.0
    range_12h = h_12h - l_12h
    
    # Camarilla levels (standard multipliers)
    r4_12h = pp_12h + range_12h * 1.1 / 2
    r3_12h = pp_12h + range_12h * 1.1 / 4
    s3_12h = pp_12h - range_12h * 1.1 / 4
    s4_12h = pp_12h - range_12h * 1.1 / 2
    
    # Align 12h levels to 6h timeframe (wait for 12h bar to close)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or \
           np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price rejection at S3/R3 (fade in range) - long at S3, short at R3
        # Price must touch or penetrate level and close back inside
        touch_s3 = low[i] <= s3_12h_aligned[i] and close[i] > s3_12h_aligned[i]
        touch_r3 = high[i] >= r3_12h_aligned[i] and close[i] < r3_12h_aligned[i]
        
        # Breakout at S4/R4 with volume (trend continuation)
        break_s4 = high[i] > s4_12h_aligned[i] and close[i] > s4_12h_aligned[i] and vol_confirm[i]
        break_r4 = low[i] < r4_12h_aligned[i] and close[i] < r4_12h_aligned[i] and vol_confirm[i]
        
        # Entry logic
        if touch_s3 and position != 1:
            # Long rejection at S3
            position = 1
            signals[i] = 0.25
        elif touch_r3 and position != -1:
            # Short rejection at R3
            position = -1
            signals[i] = -0.25
        elif break_s4 and position != 1:
            # Long breakout at S4 with volume
            position = 1
            signals[i] = 0.25
        elif break_r4 and position != -1:
            # Short breakout at R4 with volume
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (touch_r3 or break_r4):
            # Exit long on R3 rejection or R4 breakdown
            position = 0
            signals[i] = 0.0
        elif position == -1 and (touch_s3 or break_s4):
            # Exit short on S3 rejection or S4 breakout
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals