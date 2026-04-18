#!/usr/bin/env python3
"""
4h_KAMA_Direction_Volume_Trend_Filter
Hypothesis: KAMA adapts to market noise, providing a reliable trend filter. 
When price is above KAMA with volume confirmation, we go long; below KAMA with volume, short.
Uses 1d trend filter to avoid counter-trend trades. Designed for 4h timeframe to target 20-50 trades/year.
Works in bull by catching trends, in bear by avoiding false breaks via volume + trend filter.
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
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            sum_abs_diff = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if sum_abs_diff > 0:
                er[i] = price_change / sum_abs_diff
            else:
                er[i] = 0
    er[:10] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day EMA trend
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # warmup for KAMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_1d_4h[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_4h[i]
        
        if position == 0:
            # Long: price > KAMA with volume in uptrend
            if price > kama_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA with volume in downtrend
            elif price < kama_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA or trend reverses
            if price < kama_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or trend reverses
            if price > kama_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0