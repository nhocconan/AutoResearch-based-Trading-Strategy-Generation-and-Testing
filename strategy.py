#!/usr/bin/env python3
"""
1h Trend Reversal with 4h/1d Confluence and Volume Filter.
Long when 4h trend is up AND price > 1h EMA20 AND volume > 1.5x average.
Short when 4h trend is down AND price < 1h EMA20 AND volume > 1.5x average.
Exit when price crosses back below/above 1h EMA20.
Designed for 15-37 trades/year on 1h timeframe with strong trend-following edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(arr, period):
    """Exponential Moving Average"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result.fill(np.nan)
    if n < period:
        return result
    alpha = 2.0 / (period + 1.0)
    result[0] = arr[0]
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i-1]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    ema40_4h = ema(df_4h['close'].values, 50)
    ema40_4h_aligned = align_htf_to_ltf(prices, df_4h, ema40_4h)
    
    # Get 1d data for higher timeframe filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # 1d EMA100 for higher timeframe trend filter
    ema100_1d = ema(df_1d['close'].values, 100)
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 1h EMA20 for entry timing
    ema20_1h = ema(close, 20)
    
    # Volume filter: volume > 1.5x average (to avoid false signals)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need EMA20 (20) + volume MA (20)
    start_idx = max(20, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema40_4h_aligned[i]) or np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(ema20_1h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current EMA values
        ema40_4h_val = ema40_4h_aligned[i]
        ema100_1d_val = ema100_1d_aligned[i]
        ema20_1h_val = ema20_1h[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: 4h trend up AND 1d trend up AND price > EMA20 AND volume filter
            if (ema40_4h_val > ema40_4h[i-1] and  # 4h EMA rising
                ema100_1d_val > ema100_1d[i-1] and  # 1d EMA rising
                price_now > ema20_1h_val and 
                vol_filter):
                signals[i] = size
                position = 1
            # Short: 4h trend down AND 1d trend down AND price < EMA20 AND volume filter
            elif (ema40_4h_val < ema40_4h[i-1] and  # 4h EMA falling
                  ema100_1d_val < ema100_1d[i-1] and  # 1d EMA falling
                  price_now < ema20_1h_val and 
                  vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below EMA20
            if price_now < ema20_1h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above EMA20
            if price_now > ema20_1h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Trend_Reversal_4h1d_Confluence_Volume"
timeframe = "1h"
leverage = 1.0