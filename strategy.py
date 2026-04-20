#!/usr/bin/env python3
# 12h_1d_Camarilla_R1S1_Breakout_TrendFollow_V1
# Hypothesis: Daily Camarilla R1/S1 breakouts on 12h timeframe with 1d EMA trend filter and volume confirmation.
# Uses 1d EMA100 for trend (responsive to trend changes) and volume spike to avoid false breakouts.
# Target: 12-37 trades/year per symbol for balance between signal quality and frequency.
# Designed to work in both bull and bear markets by using trend filter to avoid counter-trend trades.

name = "12h_1d_Camarilla_R1S1_Breakout_TrendFollow_V1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 110:
        return np.zeros(n)
    
    # Calculate 1d pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate 1d EMA100 for trend filter
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Calculate volume average for spike detection (24h average)
    vol_ma_1d = pd.Series(volume).rolling(window=2, min_periods=2).mean().values  # 2*12h = 24h
    
    # Align 1d indicators to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 110  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema100_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0 * 1d average volume
        volume_spike = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: price > 1d EMA100 (uptrend) and breaks above R1 with volume
            if close[i] > ema100_1d_aligned[i] and close[i] > r1_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < 1d EMA100 (downtrend) and breaks below S1 with volume
            elif close[i] < ema100_1d_aligned[i] and close[i] < s1_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal) or trend changes
            if close[i] < s1_1d_aligned[i] or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal) or trend changes
            if close[i] > r1_1d_aligned[i] or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals