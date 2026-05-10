#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dTrend_Volume
Hypothesis: KAMA adapts to market noise, providing reliable trend direction. 
Combined with 1d EMA trend filter and volume confirmation to avoid false signals.
Works in both bull (KAMA upward) and bear (KAMA downward) markets.
Target: 75-200 total trades over 4 years (19-50/year).
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
    kama = np.full(n, np.nan)
    if n >= 10:
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=10))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros(n)
        er[9:] = change[9:] / np.maximum(volatility[9:], 1e-10)
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        # Initialize KAMA
        kama[9] = close[9]
        for i in range(10, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 10)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume from 1d
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0  # 24h/4h = 6
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: KAMA trending up + price above KAMA + 1d uptrend + volume
            if kama[i] > kama[i-1] and close[i] > kama[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down + price below KAMA + 1d downtrend + volume
            elif kama[i] < kama[i-1] and close[i] < kama[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA turns down or price below KAMA
            if kama[i] < kama[i-1] or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA turns up or price above KAMA
            if kama[i] > kama[i-1] or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals