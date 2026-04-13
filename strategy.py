#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d with volume confirmation.
# Camarilla pivots provide precise support/resistance levels for mean reversion and breakouts.
# Fade at R3/S3 (mean reversion), breakout continuation at R4/S4.
# Volume confirmation ensures institutional participation.
# Works in both bull and bear markets by adapting to price action at key levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # R2 = PP + (H - L) * 1.1/6
    # R1 = PP + (H - L) * 1.1/12
    # S1 = PP - (H - L) * 1.1/12
    # S2 = PP - (H - L) * 1.1/6
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots for each day
    pp = (high_1d + low_1d + close_1d) / 3
    r4 = pp + (high_1d - low_1d) * 1.1 / 2
    r3 = pp + (high_1d - low_1d) * 1.1 / 4
    s3 = pp - (high_1d - low_1d) * 1.1 / 4
    s4 = pp - (high_1d - low_1d) * 1.1 / 2
    
    # Align daily Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 20-period average
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r3_level = r3_aligned[i]
        r4_level = r4_aligned[i]
        s3_level = s3_aligned[i]
        s4_level = s4_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Mean reversion long at S3 (fade)
            if (price <= s3_level and 
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Mean reversion short at R3 (fade)
            elif (price >= r3_level and 
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            # Breakout long at R4 (continuation)
            elif (price >= r4_level and 
                  volume_confirm):
                position = 1
                signals[i] = position_size
            # Breakout short at S4 (continuation)
            elif (price <= s4_level and 
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R3 (take profit) or breaks below S4 (stop)
            if (price >= r3_level or 
                price < s4_level):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches S3 (take profit) or breaks above R4 (stop)
            if (price <= s3_level or 
                price > r4_level):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Camarilla_Pivot_Volume_FadeBreakout_v1"
timeframe = "6h"
leverage = 1.0