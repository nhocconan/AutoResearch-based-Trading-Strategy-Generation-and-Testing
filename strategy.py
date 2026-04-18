#!/usr/bin/env python3
"""
1h_4d_1d_Camarilla_Pivot_R1S1_Breakout_Volume
Hypothesis: Trade breakouts of 1d R1/S1 levels on 1h timeframe with volume confirmation and 4h trend bias.
Uses 4h EMA34 for trend filter and 1d Camarilla levels for structure. Target: 15-30 trades/year.
Designed for both bull/bear markets via 4h trend filter. Volume filter reduces false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla R1 and S1
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all to 1h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: current volume > 2.0 x 24-period average
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1d R1, above 4h EMA34, with volume
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema_4h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 1d S1, below 4h EMA34, with volume
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema_4h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 1d S1 or breaks below 4h EMA34
            if (close[i] < s1_1d_aligned[i] or close[i] < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price returns to 1d R1 or breaks above 4h EMA34
            if (close[i] > r1_1d_aligned[i] or close[i] > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4d_1d_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "1h"
leverage = 1.0