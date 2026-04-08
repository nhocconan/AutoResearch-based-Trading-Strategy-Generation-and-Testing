#!/usr/bin/env python3
# 1d_1w_pivot_reversal
# Hypothesis: Trade reversals at weekly pivot levels with daily price confirmation.
# Uses weekly Camarilla pivot levels (S3/S4 for long, R3/R4 for short) as key support/resistance.
# Enter on daily close beyond pivot with volume confirmation, exit on opposite pivot touch.
# Works in both bull and bear markets by fading extremes at institutional levels.
# Target: 10-25 trades/year on daily timeframe with strict pivot-based entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_pivot_reversal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4)
    # S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    range_1w = high_1w - low_1w
    r4 = close_1w + (range_1w * 1.1 / 2)
    r3 = close_1w + (range_1w * 1.1 / 4)
    s3 = close_1w - (range_1w * 1.1 / 4)
    s4 = close_1w - (range_1w * 1.1 / 2)
    
    # Align weekly pivot levels to daily timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Touch R3 or R4 (take profit) or stop below S4
            if close[i] >= r3_aligned[i] or close[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Touch S3 or S4 (take profit) or stop above R4
            if close[i] <= s3_aligned[i] or close[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Close above S4 with volume surge (bounce from deep support)
            if close[i] > s4_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: Close below R4 with volume surge (rejection at resistance)
            elif close[i] < r4_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals