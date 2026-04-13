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
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    
    # Support levels
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    
    # Align all data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 2.0x 20-period average
        volume_12h = volume[i]
        volume_ma_20_12h = volume_ma_20_1d_aligned[i] / 2  # 2x 12h periods per day
        volume_condition = volume_12h > (volume_ma_20_12h * 2.0)
        
        # Entry conditions: price near Camarilla levels with volume confirmation
        near_support = (close[i] <= s1_aligned[i] * 1.002) or (close[i] <= s2_aligned[i] * 1.002)
        near_resistance = (close[i] >= r1_aligned[i] * 0.998) or (close[i] >= r2_aligned[i] * 0.998)
        
        if position == 0:
            if near_support and volume_condition:
                position = 1
                signals[i] = position_size
            elif near_resistance and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches pivot or shows reversal signs
            if close[i] >= pivot_aligned[i] * 0.998:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches pivot or shows reversal signs
            if close[i] <= pivot_aligned[i] * 1.002:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation_v2"
timeframe = "12h"
leverage = 1.0