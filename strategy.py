#!/usr/bin/env python3
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
    
    # Daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's data)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot and range
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Key levels: S3, S4, R3, R4 (outer bands for stronger signals)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    
    # Daily volume spike detector
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_ratio = volume_1d / np.where(volume_ma_20 == 0, 1, volume_ma_20)
    
    # Align all to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(volume_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: volume > 2x 20-day average
        volume_spike = volume_ratio_aligned[i] > 2.0
        
        # Entry conditions: price at outer S3/S4 or R3/R4 with volume spike
        at_s3_s4 = (close[i] <= s3_aligned[i] * 1.005) or (close[i] <= s4_aligned[i] * 1.005)
        at_r3_r4 = (close[i] >= r3_aligned[i] * 0.995) or (close[i] >= r4_aligned[i] * 0.995)
        
        if position == 0:
            if at_s3_s4 and volume_spike:
                position = 1
                signals[i] = position_size
            elif at_r3_r4 and volume_spike:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches S3/S4 or shows rejection
            if close[i] <= s3_aligned[i] * 1.005:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches R3/R4 or shows rejection
            if close[i] >= r3_aligned[i] * 0.995:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Outer_Bands_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0