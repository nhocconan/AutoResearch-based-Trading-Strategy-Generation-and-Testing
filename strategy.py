#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (from 1d) + volume spike + choppiness regime filter
# Long when price touches or breaks above H3 pivot with volume >2x 20-period average and CHOP > 61.8 (ranging market)
# Short when price touches or breaks below L3 pivot with volume >2x 20-period average and CHOP > 61.8
# Exit when price reaches H4/L4 or crosses back below/above H3/L3 respectively
# Uses 1d Camarilla levels for structure, volume confirmation for momentum, chop filter to avoid trending markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    H4 = close_1d + range_1d * 1.1 / 2
    H3 = close_1d + range_1d * 1.1 / 4
    H2 = close_1d + range_1d * 1.1 / 6
    H1 = close_1d + range_1d * 1.1 / 12
    
    L4 = close_1d - range_1d * 1.1 / 2
    L3 = close_1d - range_1d * 1.1 / 4
    L2 = close_1d - range_1d * 1.1 / 6
    L1 = close_1d - range_1d * 1.1 / 12
    
    # Calculate 20-period volume average on 1d
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR) / (n * (max(high) - min(low)))) / log10(n)
    atr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    atr_1d[0] = high_1d[0] - low_1d[0]  # first value
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (14 * (max_high - min_low))) / np.log10(14)
    chop_raw = np.where((max_high - min_low) == 0, 50, chop_raw)  # avoid division by zero
    
    # Align all indicators to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = volume[i]  # Current volume (use 12h volume as proxy)
        
        if position == 0:
            # Long setup: price at or above H3 with volume spike and choppy market (range bound)
            if (price >= H3_aligned[i] and 
                vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike
                chop_aligned[i] > 61.8):                       # Choppy/ranging market
                position = 1
                signals[i] = position_size
            # Short setup: price at or below L3 with volume spike and choppy market
            elif (price <= L3_aligned[i] and 
                  vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike
                  chop_aligned[i] > 61.8):                       # Choppy/ranging market
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H4 or drops back below H3
            if price >= H4_aligned[i] or price < H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L4 or rises back above L3
            if price <= L4_aligned[i] or price > L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_H3L3_Volume_Chop"
timeframe = "12h"
leverage = 1.0