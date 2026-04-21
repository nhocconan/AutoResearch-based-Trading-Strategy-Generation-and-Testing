# 6h_Donchian_20_1dPivot_Structure_V1
# Hypothesis: Donchian(20) breakout on 6h filtered by daily pivot structure (R1/S1, R2/S2) and volume confirmation.
# Uses higher timeframe pivot levels as structural support/resistance to filter breakouts.
# Works in bull markets (breakouts continue) and bear markets (fades at pivot levels) by only taking
# breakouts aligned with pivot structure (long above R1, short below S1) with volume filter.
# Target: 50-150 total trades over 4 years = 12-37/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivots(high, low, close):
    """Calculate classic pivot points: P, R1, S1, R2, S2"""
    P = (high + low + close) / 3
    R1 = 2 * P - low
    S1 = 2 * P - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    return P, R1, S1, R2, S2

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data once for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    P_1d, R1_1d, S1_1d, R2_1d, S2_1d = calculate_pivots(high_1d, low_1d, close_1d)
    
    # Align pivot levels to 6h timeframe
    P_1d_aligned = align_htf_to_ltf(prices, df_1d, P_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    
    # 6h Donchian(20) channels
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Donchian upper (20-period high) and lower (20-period low)
    donchian_high = np.full_like(close_6h, np.nan)
    donchian_low = np.full_like(close_6h, np.nan)
    
    for i in range(len(close_6h)):
        start = max(0, i - 19)
        donchian_high[i] = np.max(high_6h[start:i+1])
        donchian_low[i] = np.min(low_6h[start:i+1])
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_6h = prices['volume'].values
    vol_ma = np.full_like(volume_6h, np.nan)
    for i in range(len(volume_6h)):
        if i >= 19:
            vol_ma[i] = np.mean(volume_6h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_filter = vol > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + above daily R1
            if (price > donchian_high[i] and volume_filter and 
                price > R1_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + below daily S1
            elif (price < donchian_low[i] and volume_filter and 
                  price < S1_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or drops below daily S1
            if price < donchian_low[i] or price < S1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or rises above daily R1
            if price > donchian_high[i] or price > R1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_1dPivot_Structure_V1"
timeframe = "6h"
leverage = 1.0