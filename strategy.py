#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_R3S3_Fade_R4S4_Breakout_Volume_Spike
Hypothesis: On 6h timeframe, fade (counter-trend) at daily Camarilla R3/S3 levels with volume confirmation,
and continue (trend-following) at R4/S4 breaks. This combines mean reversion at strong daily S/R
with breakout momentum when levels are decisively broken. Works in both bull (breakouts) and bear (fades).
Uses volume spike to avoid false signals. Targets 15-35 trades/year for low fee drag.
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
    
    # Get 1d data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Calculate Camarilla levels for each day
    # R3 = Close + 1.1*(High-Low)/4
    # S3 = Close - 1.1*(High-Low)/4
    # R4 = Close + 1.1*(High-Low)/2
    # S4 = Close - 1.1*(High-Low)/2
    camarilla_range = (high_1d - low_1d)
    r3_level = close_1d + (1.1 * camarilla_range) / 4
    s3_level = close_1d - (1.1 * camarilla_range) / 4
    r4_level = close_1d + (1.1 * camarilla_range) / 2
    s4_level = close_1d - (1.1 * camarilla_range) / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_level)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_level)
    
    # Volume spike detection: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Fade at R3/S3: short at R3 with volume spike, long at S3 with volume spike
            if price > r3 and vol_spike:
                signals[i] = -0.25
                position = -1
            elif price < s3 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Breakout at R4/S4: long at R4 break, short at S4 break
            elif price > r4 and vol_spike:
                signals[i] = 0.25
                position = 1
            elif price < s4 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit long: price returns to S3 or breaks below S4 (failed breakout)
            if price < s3 or price < s4:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit short: price returns to R3 or breaks above R4 (failed breakdown)
            if price > r3 or price > r4:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_Pivot_R3S3_Fade_R4S4_Breakout_Volume_Spike"
timeframe = "6h"
leverage = 1.0