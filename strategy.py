#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_TrendFilter
Hypothesis: Donchian channel (20-period high/low) breakouts on 4h timeframe with volume confirmation and trend filter (12h EMA34). 
Enters long when price breaks above upper band with above-average volume and 12h uptrend; enters short when price breaks below lower band with volume and 12h downtrend. 
Uses volatility-based position sizing (0.25) to limit drawdown. Designed for low trade frequency (<50/year) to minimize fee impact while capturing trends in both bull and bear markets via trend filter.
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
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # 12h EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema34_12h[i] = close_12h[i] * alpha + ema34_12h[i-1] * (1 - alpha)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with volume spike and 12h uptrend
            if (close[i] > upper[i] and vol_spike[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume spike and 12h downtrend
            elif (close[i] < lower[i] and vol_spike[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below lower band or 12h trend turns down
            if (close[i] < lower[i] or close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above upper band or 12h trend turns up
            if (close[i] > upper[i] or close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0