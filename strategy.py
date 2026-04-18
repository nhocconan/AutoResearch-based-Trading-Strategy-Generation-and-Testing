#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filtered
Hypothesis: 4-hour Donchian channel breakouts with volume confirmation and 1-day trend filter work across bull and bear markets.
The strategy captures breakouts from volatility contractions, uses volume to avoid false breakouts, and filters with daily trend.
Target: 20-40 trades/year on 4h timeframe with disciplined entry conditions.
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
    
    # Calculate 4-hour Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 1-day trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema34_1d[i] = np.mean(close_1d[0:35])
        else:
            ema34_1d[i] = close_1d[i] * (2/(34+1)) + ema34_1d[i-1] * (1 - 2/(34+1))
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume and daily uptrend
            if (close[i] > donchian_high[i] and vol_confirm[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and daily downtrend
            elif (close[i] < donchian_low[i] and vol_confirm[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Donchian low or daily trend turns down
            if (close[i] < donchian_low[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high or daily trend turns up
            if (close[i] > donchian_high[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_Filtered"
timeframe = "4h"
leverage = 1.0