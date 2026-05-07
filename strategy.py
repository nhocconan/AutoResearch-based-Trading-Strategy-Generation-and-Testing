#!/usr/bin/env python3
# 1D_KAMA_1WTrend_Volume_Signal
# Hypothesis: Daily KAMA direction (trend) combined with weekly EMA trend filter and volume confirmation.
# KAMA adapts to market noise, reducing whipsaw in sideways markets. Weekly EMA ensures alignment with higher timeframe trend.
# Volume filter (1.5x average) confirms institutional participation. Designed for low trade frequency (<25/year) to minimize fee drag.
# Works in bull markets via trend following and in bear markets via reduced false signals during chop.

name = "1D_KAMA_1WTrend_Volume_Signal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on daily close
    # Parameters: ER length=10, Fast=2, Slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper KAMA calculation
    er_length = 10
    fast_sc = 2
    slow_sc = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(er_length, len(close)):
        if volatility[i] != 0:
            er[i] = np.abs(close[i] - close[i-er_length]) / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    sc = np.where(np.isnan(sc), 0, sc)
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure we have EMA20, KAMA, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (bullish bias), price above weekly EMA20 (uptrend), volume spike
            if (close[i] > kama[i] and 
                close[i] > ema20_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (bearish bias), price below weekly EMA20 (downtrend), volume spike
            elif (close[i] < kama[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA (trend change)
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA (trend change)
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals