#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeConfirm_v1
Hypothesis: 6h Donchian(20) breakout in direction of weekly Camarilla pivot (R3/S3) with volume confirmation captures swing continuations in both bull and bear markets. Weekly pivot defines major support/resistance; Donchian breakout signals momentum; volume confirms validity. Target: 75-150 total trades over 4 years (19-38/year).
"""

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
    
    # Load weekly data ONCE before loop for Camarilla pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels (R3, S3, R4, S4)
    # Typical price = (H+L+C)/3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    range_ = df_1w['high'] - df_1w['low']
    # Camarilla levels
    r3 = typical_price + range_ * 1.1 / 4
    s3 = typical_price - range_ * 1.1 / 4
    r4 = typical_price + range_ * 1.1 / 2
    s4 = typical_price - range_ * 1.1 / 2
    
    # Align weekly pivot levels to 6h timeframe (no extra delay needed for pivot points)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    
    # 6h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection on 6h (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long logic: price breaks above Donchian high with volume spike + above weekly S3 (bullish bias)
        if high[i] > highest_high[i] and volume_spike[i] and close[i] > s3_aligned[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below Donchian low with volume spike + below weekly R3 (bearish bias)
        elif low[i] < lowest_low[i] and volume_spike[i] and close[i] < r3_aligned[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Donchian level or loses weekly bias
        elif position == 1 and (low[i] < lowest_low[i] or close[i] < s3_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (high[i] > highest_high[i] or close[i] > r3_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0