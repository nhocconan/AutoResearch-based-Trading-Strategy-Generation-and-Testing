#!/usr/bin/env python3
"""
4h_12h_camarilla_breakout_volume_v1
Hypothesis: 4-hour strategy using 12-hour Camarilla pivot levels with volume confirmation. 
Trades breakouts above/below daily pivot-based resistance/support levels only when accompanied by volume spikes.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by trading with momentum.
Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.
"""

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
    volume = prices['volume'].values
    
    # Get 12-hour data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # Formula: Based on previous day's high, low, close
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Calculate ranges
    range_12h = h_12h - l_12h
    
    # Camarilla levels (using previous bar to avoid look-ahead)
    r1_12h = c_12h + range_12h * 1.1 / 12
    r2_12h = c_12h + range_12h * 1.1 / 6
    r3_12h = c_12h + range_12h * 1.1 / 4
    r4_12h = c_12h + range_12h * 1.1 / 2
    s1_12h = c_12h - range_12h * 1.1 / 12
    s2_12h = c_12h - range_12h * 1.1 / 6
    s3_12h = c_12h - range_12h * 1.1 / 4
    s4_12h = c_12h - range_12h * 1.1 / 2
    
    # Align to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Volume spike detector (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R1 with volume spike
        if close[i] > r1_12h_aligned[i] and volume_spike[i]:
            signals[i] = 0.25
        # Short breakdown: price breaks below S1 with volume spike
        elif close[i] < s1_12h_aligned[i] and volume_spike[i]:
            signals[i] = -0.25
        else:
            # Hold flat or previous signal (minimize changes)
            signals[i] = 0.0
    
    return signals

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0