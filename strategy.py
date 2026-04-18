#!/usr/bin/env python3
"""
4h_KAMA_Trend_R1S1_Breakout_Volume
Hypothesis: Combines 1h KAMA trend direction with 1d R1/S1 breakouts and volume confirmation.
KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends.
Breakouts of daily R1/S1 in the direction of hourly trend provide high-probability entries.
Volume confirmation filters low-momentum breakouts. Designed for both bull and bear markets.
Target: 20-40 trades/year on 4h.
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
    
    # Get 1h data for KAMA trend
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1h, prepend=close_1h[0]))
    volatility = np.sum(np.abs(np.diff(close_1h)), axis=0)
    er = np.zeros_like(close_1h)
    er[1:] = change[1:] / (volatility[1:] + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1h)
    kama[0] = close_1h[0]
    for i in range(1, len(close_1h)):
        kama[i] = kama[i-1] + sc[i] * (close_1h[i] - kama[i-1])
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla R1 and S1
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1h, kama)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1d R1 and above 1h KAMA, with volume
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > kama_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 1d S1 and below 1h KAMA, with volume
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < kama_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 1h KAMA or breaks below 1d S1
            if (not np.isnan(kama_aligned[i]) and close[i] < kama_aligned[i]) or \
               (not np.isnan(s1_1d_aligned[i]) and close[i] < s1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 1h KAMA or breaks above 1d R1
            if (not np.isnan(kama_aligned[i]) and close[i] > kama_aligned[i]) or \
               (not np.isnan(r1_1d_aligned[i]) and close[i] > r1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0