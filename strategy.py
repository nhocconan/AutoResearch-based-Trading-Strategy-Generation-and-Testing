#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: Dynamic position sizing based on ATR-normalized distance from Camarilla levels.
In bull markets: long at R1 breakout, size increases with momentum.
In bear markets: short at S1 breakdown, size increases with momentum.
Uses 1d EMA34 trend filter and volume > 2x average for confirmation.
Target: 20-50 total trades over 4 years (~5-12/year) to minimize fee drag.
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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r1 = np.zeros(len(close_4h))
    camarilla_s1 = np.zeros(len(close_4h))
    for i in range(len(close_4h)):
        if high_4h[i] == low_4h[i]:
            camarilla_r1[i] = close_4h[i]
            camarilla_s1[i] = close_4h[i]
        else:
            camarilla_r1[i] = close_4h[i] + (high_4h[i] - low_4h[i]) * 1.1 / 12
            camarilla_s1[i] = close_4h[i] - (high_4h[i] - low_4h[i]) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 4h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ATR for volatility measurement and position sizing
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(atr_period, vol_ma_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA34
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Dynamic position sizing based on ATR-normalized distance from level
            if uptrend and volume_confirmation and price > r1_aligned[i]:
                # Long: size increases with momentum (price above R1)
                distance = (price - r1_aligned[i]) / atr[i]
                size = min(0.30, 0.10 + 0.05 * distance)  # Base 10% + up to 20% more
                signals[i] = size
                position = 1
            elif downtrend and volume_confirmation and price < s1_aligned[i]:
                # Short: size increases with momentum (price below S1)
                distance = (s1_aligned[i] - price) / atr[i]
                size = min(0.30, 0.10 + 0.05 * distance)  # Base 10% + up to 20% more
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below S1 or trend reverses
            if price < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns above R1 or trend reverses
            if price > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0