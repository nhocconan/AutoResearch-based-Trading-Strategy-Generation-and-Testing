#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla Pivot Reversal with Daily Volume Spike
# Hypothesis: Camarilla pivot levels act as strong support/resistance; price reversals from these levels with volume confirmation capture institutional flow. Works in bull via bounces from support, in bear via rejections from resistance. Volume spike filters for institutional participation.
# Target: 12-37 trades/year to minimize fee drag.
name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    r4 = pp + ((high_1d - low_1d) * 1.500)
    r3 = pp + ((high_1d - low_1d) * 1.250)
    r2 = pp + ((high_1d - low_1d) * 1.166)
    r1 = pp + ((high_1d - low_1d) * 1.083)
    s1 = pp - ((high_1d - low_1d) * 1.083)
    s2 = pp - ((high_1d - low_1d) * 1.166)
    s3 = pp - ((high_1d - low_1d) * 1.250)
    s4 = pp - ((high_1d - low_1d) * 1.500)
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S1 (stop loss) or reaches R1 (take profit)
            if close[i] < s1_aligned[i] or close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above R1 (stop loss) or reaches S1 (take profit)
            if close[i] > r1_aligned[i] or close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price crosses above S1 with volume (bounce from support)
            if close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below R1 with volume (rejection from resistance)
            elif close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals