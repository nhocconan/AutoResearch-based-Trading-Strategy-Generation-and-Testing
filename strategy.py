#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filter_Volume
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Combined with volume confirmation and position sizing discipline, it avoids whipsaws and captures sustained moves.
Targets 20-40 trades/year by requiring trend alignment and volume expansion.
"""

name = "4h_KAMA_Trend_Filter_Volume"
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
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # needs correction
    
    # Correct calculation of volatility over ER period
    lookback = 10
    er = np.zeros_like(close_1d)
    for i in range(lookback, len(close_1d)):
        direction = np.abs(close_1d[i] - close_1d[i - lookback])
        volatility = np.sum(np.abs(np.diff(close_1d[i - lookback:i + 1])))
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d average volume for volume filter
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (30) and volume average (20)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1d)
        uptrend_1d = close[i] > kama_aligned[i]
        downtrend_1d = close[i] < kama_aligned[i]
        
        # Volume filter: current 4h volume > 1.5x average 1d volume (scaled)
        vol_4h = volume[i]
        # Scale 1d volume to 4h equivalent (1d = 6x 4h)
        vol_4h_equiv = vol_avg_1d_aligned[i] / 6.0
        volume_filter = vol_4h > vol_4h_equiv * 1.5
        
        if position == 0:
            # Long entry: price above KAMA (uptrend) + volume participation
            if uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA (downtrend) + volume participation
            elif downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or volume dries up
            if not uptrend_1d:  # or volume_filter fails
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or volume dries up
            if not downtrend_1d:  # or volume_filter fails
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals