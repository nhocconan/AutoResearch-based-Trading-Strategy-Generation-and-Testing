#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h Camarilla Pivot + Volume Spike
# Uses 6h primary timeframe for Williams %R (14) mean reversion signals
# 12h Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) provide institutional reference
# Volume confirmation (2.0x 20-period average) ensures strong participation
# Discrete position sizing (0.25) controls fee drag
# Target: 80-160 total trades over 4 years (20-40/year) for 6h timeframe
# Williams %R identifies overbought/oversold conditions, Camarilla adds structure, volume confirms conviction
# Works in both bull and bear markets by fading at R3/S3 and breaking out at R4/S4

name = "6h_WilliamsR_12hCamarilla_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # Range = H - L
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    R3_12h = pivot_12h + range_12h * 1.1 / 2
    S3_12h = pivot_12h - range_12h * 1.1 / 2
    R4_12h = pivot_12h + range_12h * 1.1
    S4_12h = pivot_12h - range_12h * 1.1
    
    # Align Camarilla levels to 6h timeframe (wait for 12h bar close)
    R3_12h_aligned = align_htf_to_ltf(prices, df_12h, R3_12h)
    S3_12h_aligned = align_htf_to_ltf(prices, df_12h, S3_12h)
    R4_12h_aligned = align_htf_to_ltf(prices, df_12h, R4_12h)
    S4_12h_aligned = align_htf_to_ltf(prices, df_12h, S4_12h)
    
    # Calculate 6h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(R3_12h_aligned[i]) or np.isnan(S3_12h_aligned[i]) or 
            np.isnan(R4_12h_aligned[i]) or np.isnan(S4_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Fade at R3/S3: short at R3, long at S3 (mean reversion)
            fade_short = close[i] > R3_12h_aligned[i] and williams_r[i] > -20  # Overbought at R3
            fade_long = close[i] < S3_12h_aligned[i] and williams_r[i] < -80   # Oversold at S3
            
            # Breakout at R4/S4: long at R4 break, short at S4 break (continuation)
            breakout_long = close[i] > R4_12h_aligned[i] and williams_r[i] > -50  # Bullish momentum
            breakout_short = close[i] < S4_12h_aligned[i] and williams_r[i] < -50  # Bearish momentum
            
            if (fade_long or breakout_long) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif (fade_short or breakout_short) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to pivot or Williams %R reverses
            if close[i] < pivot_12h_aligned[i] or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to pivot or Williams %R reverses
            if close[i] > pivot_12h_aligned[i] or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals