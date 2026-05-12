#!/usr/bin/env python3
name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate KAMA on weekly close
    kama_1w = calculate_kama(close_1w, 30, 2, 30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily KAMA for direction
    kama_daily = calculate_kama(close, 30, 2, 30)
    
    # Daily volume spike (current volume > 1.5x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure KAMA has enough data
    
    for i in range(start_idx, n):
        # Skip if KAMA not ready
        if np.isnan(kama_1w_aligned[i]) or np.isnan(kama_daily[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above daily KAMA AND weekly KAMA trending up + volume spike
            if (close[i] > kama_daily[i] and 
                kama_1w_aligned[i] > kama_1w_aligned[i-1] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below daily KAMA AND weekly KAMA trending down + volume spike
            elif (close[i] < kama_daily[i] and 
                  kama_1w_aligned[i] < kama_1w_aligned[i-1] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below daily KAMA
            if close[i] < kama_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above daily KAMA
            if close[i] > kama_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_kama(price, er_length, fast_sc, slow_sc):
    """Calculate Kaufman Adaptive Moving Average"""
    price = np.asarray(price)
    n = len(price)
    kama = np.full(n, np.nan)
    
    if n < er_length:
        return kama
    
    # Efficiency Ratio
    change = np.abs(np.diff(price, n=er_length))
    volatility = np.sum(np.abs(np.diff(price)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Initialize KAMA
    kama[er_length-1] = price[er_length-1]
    
    # Calculate KAMA
    for i in range(er_length, n):
        kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
    
    return kama