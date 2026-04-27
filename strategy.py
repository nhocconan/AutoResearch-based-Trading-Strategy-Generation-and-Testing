#!/usr/bin/env python3
"""
1d_KAMA_Trend_1wTrend_Volume
Long when KAMA crosses above price + 1w trend up + volume spike.
Short when KAMA crosses below price + 1w trend down + volume spike.
Exit when KAMA crosses back or 1w trend reverses.
Designed for low frequency (10-25 trades/year) to minimize fee drag.
Uses Kaufman Adaptive Moving Average for trend following and 1w EMA for higher timeframe filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.empty_like(close_1w, dtype=np.float64)
    ema_1w.fill(np.nan)
    for i in range(33, len(close_1w)):
        ema_1w[i] = np.mean(close_1w[i-33:i+1])  # Simple MA for EMA approximation
    
    # Align 1w EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate KAMA (10) on 1d
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=np.float64), where=volatility!=0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # Start with close at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # Volume filter: volume > 1.5x average (to avoid false breakouts)
    vol_ma_20 = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA (10) + volume MA (20) + 1w EMA (34)
    start_idx = max(10, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        kama_now = kama[i]
        kama_prev = kama[i-1]
        trend_1w = ema_1w_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Bull: KAMA crosses above price + 1w trend up + volume spike
            if kama_now > price_now and kama_prev <= price_now and trend_1w > price_now and vol_filter:
                signals[i] = size
                position = 1
            # Bear: KAMA crosses below price + 1w trend down + volume spike
            elif kama_now < price_now and kama_prev >= price_now and trend_1w < price_now and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA crosses below price or 1w trend turns down
            if kama_now < price_now or trend_1w < price_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA crosses above price or 1w trend turns up
            if kama_now > price_now or trend_1w > price_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0