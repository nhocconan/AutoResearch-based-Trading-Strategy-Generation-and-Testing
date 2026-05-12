#!/usr/bin/env python3
name = "6h_KAMA_AdaptiveTrend_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d KAMA for trend filter
    # KAMA parameters: ER period=10, fast=2, slow=30
    price_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    direction = np.abs(np.diff(close_1d, n=10, prepend=close_1d[:10]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = np.square(er * (2/(2+1) - 2/(30+1)) + 2/(30+1))
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # 6h KAMA for entry signal
    price_change_6h = np.abs(np.diff(close, prepend=close[0]))
    direction_6h = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility_6h = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    er_6h = np.where(volatility_6h > 0, direction_6h / volatility_6h, 0)
    sc_6h = np.square(er_6h * (2/(2+1) - 2/(30+1)) + 2/(30+1))
    kama_6h = np.zeros_like(close)
    kama_6h[0] = close[0]
    for i in range(1, len(close)):
        kama_6h[i] = kama_6h[i-1] + sc_6h[i] * (close[i] - kama_6h[i-1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    # Price position relative to KAMA
    price_above_kama = close > kama_6h
    price_below_kama = close < kama_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(kama_6h[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above 6h KAMA + above 1d KAMA + volume filter
            if price_above_kama[i] and close[i] > kama_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below 6h KAMA + below 1d KAMA + volume filter
            elif price_below_kama[i] and close[i] < kama_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 6h KAMA or below 1d KAMA
            if price_below_kama[i] or close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 6h KAMA or above 1d KAMA
            if price_above_kama[i] or close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals