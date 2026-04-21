#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_TrendFilter
Hypothesis: Use 1d Camarilla pivot levels (R1/S1) for entry, 4h EMA50 for trend filter, 
and volume spike for confirmation. Enter long when price crosses above R1 in uptrend,
short when price crosses below S1 in downtrend. Exit on trend reversal. 
Designed for 4h timeframe to target 20-40 trades/year with low frequency and high win rate.
Works in bull markets by buying strength at R1 and in bear markets by selling weakness at S1.
"""

import numpy as np
import pandas as pd
from mtf_data import get_ctf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = calculate_ema(close_4h, 50)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
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
            # Long conditions: uptrend + price crosses above R1 + volume
            if (price > ema50_4h_aligned[i] and 
                price > r1_aligned[i] and 
                prices['close'].iloc[i-1] <= r1_aligned[i] and  # crossed above this bar
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: downtrend + price crosses below S1 + volume
            elif (price < ema50_4h_aligned[i] and 
                  price < s1_aligned[i] and 
                  prices['close'].iloc[i-1] >= s1_aligned[i] and  # crossed below this bar
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal (price below EMA50)
            if price < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal (price above EMA50)
            if price > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0