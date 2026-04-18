#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_Volume_V1
Hypothesis: Daily KAMA direction with 1-week EMA20 trend filter and volume confirmation.
KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends.
Designed for low trade frequency (<25/year) with strong performance in both bull and bear markets.
Uses proven EMA20 period and volume threshold (1.5x) to avoid overtrading while maintaining edge.
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
    
    # Calculate 1-day KAMA
    er_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    
    # Avoid division by zero
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i-er_period]):
            kama[i] = kama[i-1] + sc[i-er_period] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 1-week EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema20_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        if i == 20:
            ema20_1w[i] = np.mean(close_1w[0:21])
        else:
            k = 2 / (20 + 1)
            ema20_1w[i] = close_1w[i] * k + ema20_1w[i-1] * (1 - k)
    
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period + 1, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume spike and 1-week uptrend
            if (close[i] > kama[i] and vol_spike[i] and 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume spike and 1-week downtrend
            elif (close[i] < kama[i] and vol_spike[i] and 
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or 1-week trend turns down
            if (close[i] < kama[i] or close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or 1-week trend turns up
            if (close[i] > kama[i] or close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Filter_Volume_V1"
timeframe = "1d"
leverage = 1.0