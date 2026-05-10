#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dTrend_Volume
Hypothesis: KAMA adapts to market noise, providing smooth trend direction. 
In trending markets, KAMA follows price closely; in ranging markets, it flattens. 
Combined with 1d EMA34 trend filter and volume confirmation, this filters false signals.
Works in bull (KAMA up + price above) and bear (KAMA down + price below). 
Target: 20-50 total trades over 4 years (5-12/year) to avoid fee drag.
"""

name = "4h_KAMA_Direction_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # KAMA on 4h data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Fix shapes: change has length n-10, volatility has length n-1
    er = np.full(n, np.nan)
    if n >= 11:
        # volatility needs to be computed for windows of 10
        vol_window = np.full(n, np.nan)
        for i in range(9, n):
            vol_window[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
        change_window = np.full(n, np.nan)
        for i in range(10, n):
            change_window[i] = np.abs(close[i] - close[i-10])
        # Avoid division by zero
        er[10:] = change_window[10:] / np.where(vol_window[10:] == 0, 1, vol_window[10:])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA2
    slow_sc = 2 / (30 + 1)  # EMA30
    sc = np.full(n, np.nan)
    if n >= 10:
        sc[10:] = (er[10:] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.full(n, np.nan)
    if n >= 11:
        kama[10] = close[10]  # start with close
        for i in range(11, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume approximation: 4h volume from 1d (24h/4h = 6)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: KAMA rising (trend up) + price above KAMA + 1d uptrend + volume
            if kama[i] > kama[i-1] and close[i] > kama[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (trend down) + price below KAMA + 1d downtrend + volume
            elif kama[i] < kama[i-1] and close[i] < kama[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA falls or price crosses below KAMA
            if kama[i] < kama[i-1] or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA rises or price crosses above KAMA
            if kama[i] > kama[i-1] or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals