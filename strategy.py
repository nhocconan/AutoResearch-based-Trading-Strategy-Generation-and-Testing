#!/usr/bin/env python3
# 4h_1d_Pivot_R1S1_Breakout_TrendFollow_VolumeFilter
# Hypothesis: Daily pivot R1/S1 breakouts on 4h timeframe with trend confirmation and volume spike.
# Uses daily pivot levels for institutional support/resistance, EMA50 for trend filter, and volume spike to confirm breakout strength.
# Designed to work in both bull (breakouts with volume) and bear (reversals at pivot levels) markets.
# Target: 20-30 trades/year per symbol to avoid fee drag while capturing meaningful moves.

name = "4h_1d_Pivot_R1S1_Breakout_TrendFollow_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average for spike detection (20-period = ~5 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0 * 20-period average volume
        volume_spike = volume[i] > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Long: price > daily EMA50 (uptrend) and breaks above R1 with volume spike
            if close[i] > ema50_1d_aligned[i] and close[i] > r1_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < daily EMA50 (downtrend) and breaks below S1 with volume spike
            elif close[i] < ema50_1d_aligned[i] and close[i] < s1_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal) or trend changes to down
            if close[i] < s1_1d_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal) or trend changes to up
            if close[i] > r1_1d_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals