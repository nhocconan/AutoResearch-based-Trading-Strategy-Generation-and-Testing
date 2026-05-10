#!/usr/bin/env python3
"""
1d_KAMA_1W_Trend_Volume
Hypothesis: On 1d timeframe, use KAMA (adaptive moving average) for trend direction filtered by 1w KAMA trend and volume spikes. This captures trending moves while avoiding choppy periods. The strategy works in both bull and bear markets by following the higher timeframe trend and using volume confirmation to filter false signals. Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag.
"""

name = "1d_KAMA_1W_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate KAMA(10) for 1w
    kama_1w = calculate_kama(close_1w, 10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # 1d data for KAMA, volume and price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10) for 1d
    kama_1d = calculate_kama(close, 10)
    
    # Volume spike filter: current volume > 2.0x 20-day average
    vol_ma20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma20[:10] = vol_ma20[10]
    vol_ma20[-10:] = vol_ma20[-11]
    volume_filter = volume > vol_ma20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_1w_aligned[i]) or 
            np.isnan(kama_1d[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1w KAMA
        uptrend_1w = close[i] > kama_1w_aligned[i]
        downtrend_1w = close[i] < kama_1w_aligned[i]
        
        if position == 0:
            # Long: price above 1w KAMA (uptrend) with volume spike
            if uptrend_1w and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below 1w KAMA (downtrend) with volume spike
            elif downtrend_1w and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1w KAMA
            if close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1w KAMA
            if close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_kama(close, length):
    """Calculate Kaufman's Adaptive Moving Average"""
    if len(close) < length:
        return np.full_like(close, np.nan, dtype=float)
    
    # Direction: absolute change over 'length' periods
    direction = np.abs(np.diff(close, n=length, prepend=close[:length]))
    
    # Volatility: sum of absolute changes over 'length' periods
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # For rolling volatility, we need to compute it properly
    volatility = np.zeros_like(close)
    for i in range(length, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-length:i+1])))
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    
    # Efficiency Ratio
    er = direction / volatility
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama