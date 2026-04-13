#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for pivot levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate pivot points from previous day
    # Using yesterday's data for today's levels (no look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Classic pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.5x daily volume MA (adjusted for 4h)
        # 6x 4h periods per day, so daily MA/6 = approximate 4h period MA
        volume_4h_approx_ma = volume_ma_20_1d_aligned[i] / 6
        volume_condition = volume[i] > (volume_4h_approx_ma * 1.5)
        
        # Entry conditions: price touching pivot levels with volume confirmation
        # Long when price touches S1 or S2 with volume
        # Short when price touches R1 or R2 with volume
        touch_support = (low[i] <= s1_aligned[i] * 1.002) or (low[i] <= s2_aligned[i] * 1.002)  # 0.2% tolerance
        touch_resistance = (high[i] >= r1_aligned[i] * 0.998) or (high[i] >= r2_aligned[i] * 0.998)  # 0.2% tolerance
        
        if position == 0:
            if touch_support and volume_condition:
                position = 1
                signals[i] = position_size
            elif touch_resistance and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches pivot or shows rejection
            if high[i] >= pivot_aligned[i] * 0.998:  # Reached pivot level
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches pivot or shows rejection
            if low[i] <= pivot_aligned[i] * 1.002:  # Reached pivot level
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Pivot_Touch_Volume_v1"
timeframe = "4h"
leverage = 1.0