#!/usr/bin/env python3
# 1h_4h_1d_Camarilla_R1S1_Breakout_TrendFilter
# Hypothesis: 1h strategy using daily (1d) Camarilla R1/S1 levels for trend direction with 4h EMA trend filter and volume confirmation.
# Entry only during active session (08-20 UTC) to reduce noise. Target: 15-35 trades/year per symbol.
# Uses daily trend filter to avoid counter-trend trades in choppy markets.

name = "1h_4h_1d_Camarilla_R1S1_Breakout_TrendFilter"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d and 4h indicators to 1h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if outside session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > average of last 24 periods
        if i >= 24:
            vol_avg = np.mean(volume[i-24:i])
            volume_ok = volume[i] > vol_avg
        else:
            volume_ok = True  # Not enough data for volume check
        
        if position == 0:
            # Long: price > 4h EMA20 (uptrend) and breaks above daily R1 with volume
            if close[i] > ema20_4h_aligned[i] and close[i] > r1_1d_aligned[i] and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA20 (downtrend) and breaks below daily S1 with volume
            elif close[i] < ema20_4h_aligned[i] and close[i] < s1_1d_aligned[i] and volume_ok:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below daily S1 or trend changes
            if close[i] < s1_1d_aligned[i] or close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above daily R1 or trend changes
            if close[i] > r1_1d_aligned[i] or close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals