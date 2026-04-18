#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_12hTrend
Hypothesis: Donchian(20) breakouts with volume confirmation and 12h trend filter
capture strong directional moves while avoiding whipsaws. Works in both bull and bear
markets by requiring trend alignment and volume surge, reducing false breakouts.
Target: 20-40 trades/year on 4h timeframe.
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
    
    # Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    # 12h EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = np.full(len(close_12h), np.nan)
    k = 2 / (34 + 1)
    for i in range(34, len(close_12h)):
        if i == 34:
            ema34_12h[i] = np.mean(close_12h[0:35])
        else:
            ema34_12h[i] = close_12h[i] * k + ema34_12h[i-1] * (1 - k)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and 12h uptrend
            if (close[i] > donchian_high[i] and vol_spike[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and 12h downtrend
            elif (close[i] < donchian_low[i] and vol_spike[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Donchian low or 12h trend turns down
            if (close[i] < donchian_low[i] or close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high or 12h trend turns up
            if (close[i] > donchian_high[i] or close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0