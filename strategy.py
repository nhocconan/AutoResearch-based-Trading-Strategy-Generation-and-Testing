#!/usr/bin/env python3
"""
6H_Williams_R1_S1_Reversal
Hypothesis: Williams %R extreme readings (below -80 oversold, above -20 overbought) on 1d timeframe signal potential reversals, confirmed by price rejection at daily R1/S1 pivot levels and volume spike. Enter long when Williams %R crosses above -80 from below with price at or above daily S1 and volume confirmation; enter short when crosses below -20 from above with price at or below daily R1 and volume confirmation. Exit when Williams %R returns to neutral range (-50) or opposite extreme is reached. Designed for 6h timeframe to capture intraday reversals within daily extremes while limiting trades to avoid fee drag. Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets.
"""

name = "6H_Williams_R1_S1_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R and pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align Williams %R and pivot levels to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume average (24-period for 6d equivalent) for volume spike filter
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(williams_r_aligned[i-1]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 24-period average
        vol_spike = volume[i] > 2.0 * vol_ma_24[i]
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below (ending oversold) + price at/above S1 + volume spike
            if (williams_r_aligned[i-1] <= -80 and williams_r_aligned[i] > -80 and 
                close[i] >= s1_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above (ending overbought) + price at/below R1 + volume spike
            elif (williams_r_aligned[i-1] >= -20 and williams_r_aligned[i] < -20 and 
                  close[i] <= r1_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R reaches -20 (overbought) or crosses below -50 (losing momentum)
            if williams_r_aligned[i] >= -20 or williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R reaches -80 (oversold) or crosses above -50 (losing momentum)
            if williams_r_aligned[i] <= -80 or williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals