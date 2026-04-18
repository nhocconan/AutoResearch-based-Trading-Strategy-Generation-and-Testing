#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Momentum
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) for breakout entries with momentum confirmation on 4h. 
Enters long when price breaks above R1 with bullish momentum (price > EMA20), short when breaks below S1 with bearish momentum (price < EMA20). 
Volume confirmation filters weak breakouts. Designed for moderate trade frequency (~30-60/year) with strong performance in both bull and bear markets via pivot-based support/resistance.
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
    
    # EMA20 for momentum filter
    ema_period = 20
    ema = np.full(n, np.nan)
    k = 2 / (ema_period + 1)
    for i in range(ema_period, n):
        if i == ema_period:
            ema[i] = np.mean(close[i-ema_period+1:i+1])
        else:
            ema[i] = close[i] * k + ema[i-1] * (1 - k)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    h1d = df_1d['high'].values
    l1d = df_1d['low'].values
    c1d = df_1d['close'].values
    range_1d = h1d - l1d
    
    # Camarilla formulas
    r1 = c1d + (range_1d * 1.1 / 12)
    s1 = c1d - (range_1d * 1.1 / 12)
    
    # Align to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with bullish momentum and volume spike
            if close[i] > r1_aligned[i] and close[i] > ema[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with bearish momentum and volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below EMA20 or breaks below S1 (invalidates bullish setup)
            if close[i] < ema[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above EMA20 or breaks above R1 (invalidates bearish setup)
            if close[i] > ema[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Momentum"
timeframe = "4h"
leverage = 1.0