#!/usr/bin/env python3
"""
4h_Supertrend_Breakout_12hTrend_Confirmation
Hypothesis: Combine Supertrend (ATR-based trend following) from 12h with 4h price breakout above/below ATR-based channels. 
Supertrend provides robust trend direction that works in both bull and bear markets by adapting to volatility. 
Breakout confirmation ensures we enter with momentum. Volume filter adds confirmation of institutional interest.
Target: 20-40 trades per year on 4h timeframe with strong risk-adjusted returns.
"""

name = "4h_Supertrend_Breakout_12hTrend_Confirmation"
timeframe = "4h"
leverage = 1.0

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
    
    # === 12H Data for Supertrend Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Supertrend calculation (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]  # First period
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
            
        # Adjust bands
        if direction[i] == 1:
            if lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        else:
            if upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            if lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
    
    # Align 12H Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # === 4H Indicators for Entry Signals ===
    # ATR for channel calculation (same parameters as Supertrend)
    tr_4h1 = high - low
    tr_4h2 = np.abs(high - np.roll(close, 1))
    tr_4h3 = np.abs(low - np.roll(close, 1))
    tr_4h1[0] = high[0] - low[0]
    tr_4h2[0] = np.abs(high[0] - close[0])
    tr_4h3[0] = np.abs(low[0] - close[0])
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    
    atr_4h = np.zeros_like(tr_4h)
    atr_4h[atr_period-1] = np.mean(tr_4h[:atr_period])
    for i in range(atr_period, len(tr_4h)):
        atr_4h[i] = (atr_4h[i-1] * (atr_period-1) + tr_4h[i]) / atr_period
    
    # ATR-based channels (similar to Donchian but volatility-adjusted)
    upper_channel = close + (atr_4h * 1.5)
    lower_channel = close - (atr_4h * 1.5)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = np.mean(volume[:20]) if len(volume) >= 20 else volume[0]
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(30, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel AND 12h uptrend AND volume spike
            if close[i] > upper_channel[i] and direction_aligned[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND 12h downtrend AND volume spike
            elif close[i] < lower_channel[i] and direction_aligned[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below lower channel OR 12h trend turns down
            if close[i] < lower_channel[i] or direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above upper channel OR 12h trend turns up
            if close[i] > upper_channel[i] or direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals