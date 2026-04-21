#!/usr/bin/env python3
"""
4h_PivotPoint_Reversal_Volume
Hypothesis: Combine daily pivot point reversals with volume confirmation. 
Enter long when price bounces off daily support (S1/S2) with volume spike, 
enter short when price rejects daily resistance (R1/R2) with volume spike.
Works in both bull/bear markets by using mean-reversion at key daily levels.
Target 20-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d OHLC data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily pivot points (standard calculation) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance levels
    r1 = 2 * pp - low_1d
    r2 = pp + (high_1d - low_1d)
    # Support levels
    s1 = 2 * pp - high_1d
    s2 = pp - (high_1d - low_1d)
    
    # Align to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price bounces off S1 or S2 with volume spike
            # Condition: low touches/below support AND close above support
            if (vol_spike > 1.5 and 
                ((price_low := prices['low'].iloc[i]) <= s1_aligned[i] and price_close > s1_aligned[i]) or
                ((price_low := prices['low'].iloc[i]) <= s2_aligned[i] and price_close > s2_aligned[i])):
                signals[i] = 0.25
                position = 1
            # Short: price rejects R1 or R2 with volume spike
            # Condition: high touches/above resistance AND close below resistance
            elif (vol_spike > 1.5 and 
                  ((price_high := prices['high'].iloc[i]) >= r1_aligned[i] and price_close < r1_aligned[i]) or
                  ((price_high := prices['high'].iloc[i]) >= r2_aligned[i] and price_close < r2_aligned[i])):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to pivot point (mean reversion complete)
            if position == 1 and price_close >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_PivotPoint_Reversal_Volume"
timeframe = "4h"
leverage = 1.0