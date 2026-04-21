#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_TrendFilter
Hypothesis: Camarilla pivot levels (R1/S1) from 1d combined with volume spike (>1.5x avg) and 1w EMA100 trend filter provides high-probability breakout trades. The 1w EMA100 ensures we trade with the weekly trend, reducing false signals during counter-trend moves. Volume confirmation ensures breakout is backed by participation. Designed for low trade frequency (target: 20-40/year) to minimize fee drag in 4h timeframe. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # AlCamarilla levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 1w data for EMA100 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA100 for trend filter
    close_1w = df_1w['close'].values
    ema100_1w = np.zeros_like(close_1w)
    ema100_1w[0] = close_1w[0]
    alpha = 2.0 / (100 + 1)
    for i in range(1, len(close_1w)):
        ema100_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema100_1w[i-1]
    
    # Align 1w EMA100 to 4h timeframe
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if NaN in critical values
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema100_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema100 = ema100_1w_aligned[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and 1w uptrend (price > 1w EMA100)
            if price > r1_level and vol_ok and price > ema100:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume and 1w downtrend (price < 1w EMA100)
            elif price < s1_level and vol_ok and price < ema100:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls back below R1 (failed breakout) or breaks below 1w EMA100 (trend change)
            if price < r1_level or price < ema100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S1 (failed breakdown) or breaks above 1w EMA100 (trend change)
            if price > s1_level or price > ema100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0