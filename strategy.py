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
    volume = prices['volume'].values
    
    # === 1d data (HTF for Camarilla pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === Calculate Camarilla pivot levels from 1d data ===
    # P = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance and Support levels
    # R4 = Close + ((High - Low) * 1.5)
    # R3 = Close + ((High - Low) * 1.25)
    # R2 = Close + ((High - Low) * 1.166)
    # R1 = Close + ((High - Low) * 1.083)
    # S1 = Close - ((High - Low) * 1.083)
    # S2 = Close - ((High - Low) * 1.166)
    # S3 = Close - ((High - Low) * 1.25)
    # S4 = Close - ((High - Low) * 1.5)
    
    r4 = close_1d + (range_1d * 1.5)
    r3 = close_1d + (range_1d * 1.25)
    r2 = close_1d + (range_1d * 1.166)
    r1 = close_1d + (range_1d * 1.083)
    s1 = close_1d - (range_1d * 1.083)
    s2 = close_1d - (range_1d * 1.166)
    s3 = close_1d - (range_1d * 1.25)
    s4 = close_1d - (range_1d * 1.5)
    
    # Align Camarilla levels to 6h timeframe (wait for daily close)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_ratio = vol_ratio_6h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below S3 (strong support break)
            if price < s3_6h[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above R3 (strong resistance break)
            if price > r3_6h[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above R4 with volume (continuation)
            if price > r4_6h[i] and vol_ratio > 1.8:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Break below S4 with volume (continuation)
            elif price < s4_6h[i] and vol_ratio > 1.8:
                signals[i] = -0.25
                position = -1
                continue
            # LONG: Reversal from S3 (buy the dip)
            elif price < s3_6h[i] and vol_ratio > 1.5 and i >= 2:
                # Look for rejection: price below S3 but closing back above
                if close[i] > s3_6h[i] and close[i-1] <= s3_6h[i-1]:
                    signals[i] = 0.25
                    position = 1
                    continue
            # SHORT: Reversal from R3 (sell the rally)
            elif price > r3_6h[i] and vol_ratio > 1.5 and i >= 2:
                # Look for rejection: price above R3 but closing back below
                if close[i] < r3_6h[i] and close[i-1] >= r3_6h[i-1]:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Reversal_R4_S4_Breakout"
timeframe = "6h"
leverage = 1.0