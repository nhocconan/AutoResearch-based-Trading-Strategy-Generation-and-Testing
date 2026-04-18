#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA34_Volume_Momentum
Hypothesis: Uses Camarilla pivot levels (R1/S1) from daily with 12h EMA34 trend filter and volume confirmation.
Enters long when price breaks above R1 with 12h EMA34 rising and volume spike, short when breaks below S1 with EMA34 falling and volume spike.
Designed for fewer trades (~20-30/year) with strong trend capture in both bull and bear markets via institutional pivot levels.
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
    
    # Daily high/low/close for Camarilla calculation (using 1d data from mtf)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    # Camarilla R1 and S1 levels
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # EMA34 slope (rising/falling)
    ema34_slope = np.zeros_like(ema34_aligned)
    ema34_slope[1:] = ema34_aligned[1:] - ema34_aligned[:-1]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(ema34_slope[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with rising EMA34 and volume spike
            if close[i] > r1_aligned[i] and ema34_slope[i] > 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with falling EMA34 and volume spike
            elif close[i] < s1_aligned[i] and ema34_slope[i] < 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or EMA34 turns down
            if close[i] < s1_aligned[i] or ema34_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or EMA34 turns up
            if close[i] > r1_aligned[i] or ema34_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Volume_Momentum"
timeframe = "4h"
leverage = 1.0