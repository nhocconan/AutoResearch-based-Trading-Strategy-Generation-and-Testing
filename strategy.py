#!/usr/bin/env python3
# 12h_KAMA_1dTrend_Volume
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 1d timeframe to determine trend direction, filtered by 1w volume spike and ADX trend strength. KAMA adapts to market noise, reducing false signals in choppy markets. Volume confirms institutional interest, and ADX ensures we only trade in strong trends. Works in both bull and bear markets by following the higher timeframe trend.

name = "12h_KAMA_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get 1d data for KAMA trend and volume calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (10, 2, 30) on 1d close
    close_1d = df_1d['close'].values
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        change = abs(close_1d[i] - close_1d[i-10])
        vol_sum = np.sum(np.abs(close_1d[i-9:i+1] - close_1d[i-10:i]))
        er[i] = change / vol_sum if vol_sum != 0 else 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Get 1w data for ADX trend strength
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period) on 1w data
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    tr = np.zeros_like(high_1w)
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i-1], 0) if (high_1w[i] - high_1w[i-1]) > (low_1w[i-1] - low_1w[i]) else 0
        minus_dm[i] = max(low_1w[i-1] - low_1w[i], 0) if (low_1w[i-1] - low_1w[i]) > (high_1w[i] - high_1w[i-1]) else 0
        tr[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    
    atr = np.zeros_like(high_1w)
    atr[13] = np.mean(tr[1:14])
    for i in range(14, len(high_1w)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = 100 * (np.zeros_like(high_1w))
    minus_di = 100 * (np.zeros_like(high_1w))
    for i in range(14, len(high_1w)):
        plus_di[i] = 100 * (np.mean(plus_dm[i-13:i+1]) / atr[i]) if atr[i] != 0 else 0
        minus_di[i] = 100 * (np.mean(minus_dm[i-13:i+1]) / atr[i]) if atr[i] != 0 else 0
    
    dx = np.zeros_like(high_1w)
    for i in range(14, len(high_1w)):
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
    
    adx = np.zeros_like(high_1w)
    adx[27] = np.mean(dx[14:28])
    for i in range(28, len(high_1w)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align indicators to 12h timeframe
    kama_12h = align_htf_to_ltf(prices, df_1d, kama)
    adx_12h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate volume spike on 1d (20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = np.zeros_like(volume_1d)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    volume_spike = np.zeros_like(volume_1d, dtype=bool)
    for i in range(19, len(volume_1d)):
        volume_spike[i] = volume_1d[i] > (2.0 * vol_ma_20[i])
    volume_spike_12h = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_12h[i]) or np.isnan(adx_12h[i]) or 
            np.isnan(volume_spike_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA + ADX > 25 (strong trend) + volume spike
            if close[i] > kama_12h[i] and adx_12h[i] > 25 and volume_spike_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + ADX > 25 (strong trend) + volume spike
            elif close[i] < kama_12h[i] and adx_12h[i] > 25 and volume_spike_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below KAMA or ADX drops below 20 (weakening trend)
            if close[i] < kama_12h[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above KAMA or ADX drops below 20 (weakening trend)
            if close[i] > kama_12h[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals