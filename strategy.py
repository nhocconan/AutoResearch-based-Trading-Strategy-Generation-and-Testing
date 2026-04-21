#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_R1S1_Breakout_Volume_TrendFilter_V1
Hypothesis: Use 1d Camarilla pivot levels (R1/S1) and 1w trend (price > 1w EMA34) for bias.
On 4h, enter long when price breaks above S1 with volume spike and 1w uptrend.
Enter short when price breaks below R1 with volume spike and 1w downtrend.
Exit on trend reversal or price crossing the pivot point (PP).
Designed for 4h timeframe with 1d/1w filters to limit trades to ~20-50/year.
Works in bull markets by buying strength and in bear markets by selling weakness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    pp = (high + low + close) / 3
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return pp, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    pp_1d = np.zeros_like(high_1d)
    r1_1d = np.zeros_like(high_1d)
    s1_1d = np.zeros_like(high_1d)
    
    for i in range(len(high_1d)):
        pp, r1, s1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        pp_1d[i] = pp
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Align 1d levels to 4h
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = calculate_ema(close_1w, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: uptrend + break above S1 + volume
            if (price > ema34_1w_aligned[i] and 
                price > s1_1d_aligned[i] and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: downtrend + break below R1 + volume
            elif (price < ema34_1w_aligned[i] and 
                  price < r1_1d_aligned[i] and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or price crosses below PP
            if price < ema34_1w_aligned[i] or price < pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or price crosses above PP
            if price > ema34_1w_aligned[i] or price > pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_1w_Camarilla_R1S1_Breakout_Volume_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0