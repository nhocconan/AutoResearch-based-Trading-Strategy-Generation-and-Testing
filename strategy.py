#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume
Hypothesis: Combines Camarilla pivot levels (R1/S1) from 1d with EMA34 trend filter and volume confirmation.
Enters long when price breaks above R1 with EMA34 rising, short when price breaks below S1 with EMA34 falling.
Uses 4h timeframe for entries, targeting 20-40 trades/year. Designed to work in both bull and bear markets
by following the 1d trend while using Camarilla levels for precise entry/exit.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate EMA34 on 1d for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # EMA34 slope for trend direction
    ema34_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema34_aligned[i]) and not np.isnan(ema34_aligned[i-1]):
            ema34_slope[i] = ema34_aligned[i] - ema34_aligned[i-1]
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema34_aligned[i]) or np.isnan(ema34_slope[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with rising EMA34 and volume spike
            if close[i] > r1_aligned[i] and ema34_slope[i] > 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with falling EMA34 and volume spike
            elif close[i] < s1_aligned[i] and ema34_slope[i] < 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S1 or EMA34 turns down
            if close[i] < s1_aligned[i] or ema34_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R1 or EMA34 turns up
            if close[i] > r1_aligned[i] or ema34_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0