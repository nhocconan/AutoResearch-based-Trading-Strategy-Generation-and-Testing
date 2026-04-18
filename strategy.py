#!/usr/bin/env python3
"""
1d_Weekly_KAMA_Breakout_Volume
Hypothesis: KAMA adapts to market noise, providing a robust trend filter on 1-day charts.
Breakouts above/below KAMA with weekly trend confirmation and volume spikes capture
trend continuations in both bull and bear markets while avoiding whipsaws in chop.
Target: 15-25 trades/year on 1d timeframe with disciplined entry conditions.
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
    
    # Calculate 1-day KAMA (adaptive moving average)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio and KAMA calculation
    er_1d = np.full(len(close_1d), np.nan)
    kama_1d = np.full(len(close_1d), np.nan)
    
    for i in range(10, len(close_1d)):  # ER period = 10
        change = abs(close_1d[i] - close_1d[i-10])
        volatility = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
        if volatility > 0:
            er_1d[i] = change / volatility
        else:
            er_1d[i] = 0
        
        # Smoothing constants
        fast_sc = 2 / (2 + 1)   # EMA(2)
        slow_sc = 2 / (30 + 1)  # EMA(30)
        sc = (er_1d[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == 10:
            kama_1d[i] = close_1d[i]
        else:
            kama_1d[i] = kama_1d[i-1] + sc * (close_1d[i] - kama_1d[i-1])
    
    # Align 1-day KAMA to daily timeframe (same timeframe, no alignment needed)
    kama_1d_aligned = kama_1d  # Already on 1d timeframe
    
    # Calculate 1-week EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 34:
        k = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            if i == 34:
                ema34_1w[i] = np.mean(close_1w[0:35])
            else:
                ema34_1w[i] = close_1w[i] * k + ema34_1w[i-1] * (1 - k)
    
    # Align 1-week EMA to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume spike and weekly uptrend
            if (close[i] > kama_1d_aligned[i] and vol_spike[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume spike and weekly downtrend
            elif (close[i] < kama_1d_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or weekly trend turns down
            if (close[i] < kama_1d_aligned[i] or close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or weekly trend turns up
            if (close[i] > kama_1d_aligned[i] or close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_KAMA_Breakout_Volume"
timeframe = "1d"
leverage = 1.0