#!/usr/bin/env python3
"""
4h_KAMA_Trend_Reversal_1dVolatilityBreakout_v1
Hypothesis: KAMA adapts to market efficiency, providing smooth trend signals. Combine KAMA direction with 1-day volatility breakout (ATR-based) to catch trend reversals in both bull and bear markets. Volatility expansion confirms institutional participation. Target: 20-40 trades per year on 4h timeframe.
"""

name = "4h_KAMA_Trend_Reversal_1dVolatilityBreakout_v1"
timeframe = "4h"
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
    
    # === 1D Data for Volatility Breakout ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1D ATR for volatility breakout
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility breakout: today's range > 1.5x average ATR
    daily_range = high_1d - low_1d
    vol_breakout = daily_range > (atr_14 * 1.5)
    
    # Align volatility breakout to 4h timeframe
    vol_breakout_aligned = align_htf_to_ltf(prices, df_1d, vol_breakout)
    
    # === 4h KAMA Calculation ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    change = np.concatenate([[np.nan]*10, change])
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Proper volatility calculation (sum of absolute changes)
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility = np.concatenate([[np.nan]*9, volatility[9:]])  # 10-period sum
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or 
            np.isnan(vol_breakout_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA AND volatility breakout (expansion)
            if close[i] > kama[i] and vol_breakout_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA AND volatility breakout (expansion)
            elif close[i] < kama[i] and vol_breakout_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals