# 6h_Pivot_R1_S1_Breakout_Volume_RangeFilter
# Hypothesis: Fade at Camarilla R1/S1 during range-bound markets (CHOP > 61.8), breakout continuation in trending markets (CHOP < 38.2). Uses 1d pivots for structure, 6h for entry timing. Works in bull/bear via regime filter.
# Target: 12-37 trades/year (50-150 over 4 years). Size: 0.25.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Camarilla pivots and chop ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate Camarilla levels (R1, S1, R4, S4) ===
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R4 = C + (H - L) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # === Align 1d levels to 6h ===
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === Chop index (14-period) for regime detection ===
    # CHOP = 100 * log10(SUM(TR1) / (ATR * N)) / log10(N)
    # Simplified: use high-low range for TR
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14)
    chop_6h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 6h volume confirmation ===
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume / vol_ma_20_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(chop_6h[i]) or np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1 = r1_6h[i]
        s1 = s1_6h[i]
        r4 = r4_6h[i]
        s4 = s4_6h[i]
        chop = chop_6h[i]
        vol_ratio = vol_ratio_6h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price reaches R4 (take profit) or reverses below S1 in range
            if price >= r4 or (chop > 61.8 and price < s1):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price reaches S4 (take profit) or reverses above R1 in range
            if price <= s4 or (chop > 61.8 and price > r1):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine regime: chop > 61.8 = range, chop < 38.2 = trend
            if chop > 61.8:  # Range-bound: fade at R1/S1
                # LONG at S1 with rejection (price < S1 but closing back above)
                if i > 0 and close[i-1] < s1 and price > s1 and vol_ratio > 1.3:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT at R1 with rejection (price > R1 but closing back below)
                elif i > 0 and close[i-1] > r1 and price < r1 and vol_ratio > 1.3:
                    signals[i] = -0.25
                    position = -1
                    continue
            elif chop < 38.2:  # Trending: breakout continuation at R4/S4
                # LONG on break above R4 with volume
                if price > r4 and vol_ratio > 1.5:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT on break below S4 with volume
                elif price < s4 and vol_ratio > 1.5:
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

name = "6h_Pivot_R1_S1_Breakout_Volume_RangeFilter"
timeframe = "6h"
leverage = 1.0