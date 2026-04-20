#!/usr/bin/env python3
# 1h_4h_1d_Camarilla_R1S1_Breakout_Volume_TrendFilter_v1
# Hypothesis: Use 1d Camarilla pivot levels (R1/S1) with 1h breakout, volume confirmation, and 4h EMA34 trend filter.
# 1h timeframe with 4h/1d filters to reduce noise and capture directional moves. Target: 15-37 trades/year per symbol.
# Only trade breakouts aligned with 4h trend. Volume spike >2.0x average to ensure conviction.
# Designed to work in both bull and bear markets by following trend and avoiding counter-trend trades.

name = "1h_4h_1d_Camarilla_R1S1_Breakout_Volume_TrendFilter_v1"
timeframe = "1h"
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
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1, S1
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align 1d levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume spike and uptrend (4h EMA)
            if (close[i] > r1_aligned[i] and 
                volume[i] > 2.0 * volume_ma[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price breaks below S1 with volume spike and downtrend (4h EMA)
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > 2.0 * volume_ma[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend reverses (below 4h EMA)
            if close[i] < s1_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend reverses (above 4h EMA)
            if close[i] > r1_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals