#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Filtered
Hypothesis: Uses daily trend filter with Camarilla pivot breakout on 4h. 
Enters long when price breaks above R1 with daily close above daily EMA34 and volume spike.
Enters short when price breaks below S1 with daily close below daily EMA34 and volume spike.
Uses daily EMA34 for trend filter to avoid counter-trend trades, reducing false signals in chop.
Designed for 20-40 trades/year with strong trend capture in both bull and bear markets.
"""

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
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        k = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            if i == 34:
                ema34_1d[i] = np.mean(close_1d[i-34+1:i+1])
            else:
                ema34_1d[i] = close_1d[i] * k + ema34_1d[i-1] * (1 - k)
    
    # Align daily EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R1 and S1 from previous day
    R1 = close_1d_prev + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d_prev - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with daily uptrend and volume spike
            if close[i] > R1_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with daily downtrend and volume spike
            elif close[i] < S1_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or daily trend turns down
            if close[i] < S1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or daily trend turns up
            if close[i] > R1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Filtered"
timeframe = "4h"
leverage = 1.0