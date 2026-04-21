#!/usr/bin/env python3
"""
4h_PivotPoint_R1S1_Breakout_Volume_Confirmation_V2
Hypothesis: Pivot point breakout with volume confirmation and trend filter. Works in bull/bear markets by trading breakouts from daily Camarilla pivot levels (R1/S1) in direction of 12h EMA trend. Uses volume spike confirmation to avoid false breakouts. Targets 20-40 trades/year with tight entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend direction
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Load daily data once for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate volume spike confirmation (current > 1.5 * 20-period average)
    volume = prices['volume'].values
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if np.isnan(ema_12h_aligned[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and price > 12h EMA (uptrend)
            if price > r1_aligned[i] and vol_spike and price > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and price < 12h EMA (downtrend)
            elif price < s1_aligned[i] and vol_spike and price < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below pivot or trend changes
            if price < pivot_aligned[i] or price < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above pivot or trend changes
            if price > pivot_aligned[i] or price > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PivotPoint_R1S1_Breakout_Volume_Confirmation_V2"
timeframe = "4h"
leverage = 1.0