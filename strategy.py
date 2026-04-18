#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and EMA Trend Filter
Hypothesis: Donchian channel breakouts indicate institutional activity.
Volume confirmation filters false breakouts. EMA(34) filter ensures trades follow
higher timeframe trend, improving performance in both bull and bear markets.
Limited trade frequency via strict breakout conditions reduces fee drag.
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
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on daily close
    close_1d = df_1d['close'].values
    ema_34 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        multiplier = 2 / (34 + 1)
        ema_34[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34[i] = (close_1d[i] - ema_34[i-1]) * multiplier + ema_34[i-1]
    
    # Align EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Donchian channel (20-period) on 4h data
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above daily EMA
            if (high[i] > donchian_high[i] and vol_spike[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and below daily EMA
            elif (low[i] < donchian_low[i] and vol_spike[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian low or volume dies
            if low[i] < donchian_low[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian high or volume dies
            if high[i] > donchian_high[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_EMA34"
timeframe = "4h"
leverage = 1.0