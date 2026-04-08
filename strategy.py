#!/usr/bin/env python3
# 12h_kama_trend_volume_v1
# Hypothesis: Uses KAMA on 1d to establish trend, enters long/short on 12h price crossing above/below KAMA with volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
# Volume confirmation ensures conviction behind moves. Designed for 12-37 trades/year to avoid fee drag.
# Works in both bull and bear markets by following adaptive trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prev_KAMA + SC * (close - prev_KAMA)
    
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, 10))  # |close[t] - close[t-10]|
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
        if i >= 10:
            volatility[i] -= np.abs(close_1d[i-10] - close_1d[i-11])
    
    er = np.zeros_like(close_1d)
    er[9:] = change[9:] / volatility[9:]
    er[volatility == 0] = 0  # Avoid division by zero
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align 1d KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 12h volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or volume fails
            if close[i] < kama_aligned[i] or not volume_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or volume fails
            if close[i] > kama_aligned[i] or not volume_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: price crosses above KAMA
                if close[i] > kama_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price crosses below KAMA
                elif close[i] < kama_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals