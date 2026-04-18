#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout_With_Volume_and_Trend
Hypothesis: Use 12h Donchian breakout with volume confirmation and 1d EMA trend filter to capture trending moves while avoiding whipsaws. 
Long when price breaks above 12h Donchian high with volume > 1.5x average and price > 1d EMA50. 
Short when price breaks below 12h Donchian low with volume > 1.5x average and price < 1d EMA50. 
Exit on opposite Donchian break or trend reversal. 
Designed for 12h timeframe to target 15-30 trades/year with strong trend capture in both bull and bear markets.
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
    
    # Get 12h data for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channel (20 periods)
    donchian_high = np.full_like(high_12h, np.nan)
    donchian_low = np.full_like(low_12h, np.nan)
    
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    # Align Donchian channels to 12h timeframe (wait for bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = close_1d[i] * 0.04 + ema_50[i-1] * 0.96
    
    # Align EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 12h Donchian high, with volume, and above 1d EMA50
            if (close[i] > donchian_high_aligned[i] and vol_confirm[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 12h Donchian low, with volume, and below 1d EMA50
            elif (close[i] < donchian_low_aligned[i] and vol_confirm[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below 12h Donchian low or falls below 1d EMA50
            if (close[i] < donchian_low_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 12h Donchian high or rises above 1d EMA50
            if (close[i] > donchian_high_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Donchian_Breakout_With_Volume_and_Trend"
timeframe = "12h"
leverage = 1.0