#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike
Hypothesis: Breakouts of Camarilla R1/S1 levels on 12h timeframe with 1d EMA34 trend filter and volume spike (>2x average) confirmation. Designed for fewer trades (target 12-37/year) to reduce fee drag while capturing momentum in both bull and bear markets. Uses strict entry conditions to avoid overtrading.
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r1 = np.zeros(len(close_12h))
    camarilla_s1 = np.zeros(len(close_12h))
    for i in range(len(close_12h)):
        if high_12h[i] == low_12h[i]:
            camarilla_r1[i] = close_12h[i]
            camarilla_s1[i] = close_12h[i]
        else:
            camarilla_r1[i] = close_12h[i] + (high_12h[i] - low_12h[i]) * 1.1 / 12
            camarilla_s1[i] = close_12h[i] - (high_12h[i] - low_12h[i]) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (already aligned via df_12h index)
    r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close with proper smoothing
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_1d[33] = np.mean(close_1d[:34])  # Simple average for first value
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 12h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ATR for volatility measurement (14-period)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Volume confirmation (20-period average)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: ensure all indicators are valid
    start_idx = max(atr_period, vol_ma_period, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price relative to 1d EMA34
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long entry: price breaks above R1 with uptrend and volume confirmation
            if uptrend and volume_confirmation and price > r1_aligned[i]:
                signals[i] = 0.25  # 25% position
                position = 1
            # Short entry: price breaks below S1 with downtrend and volume confirmation
            elif downtrend and volume_confirmation and price < s1_aligned[i]:
                signals[i] = -0.25  # 25% short
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below S1 or trend turns down
            if price < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns above R1 or trend turns up
            if price > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0