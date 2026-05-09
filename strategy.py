#!/usr/bin/env python3
# 12h_KAMA_Trend_1wTrendFilter
# Strategy: Trade KAMA trend direction on 12h with 1w EMA200 filter
# Long when KAMA trend up and price > 1w EMA200
# Short when KAMA trend down and price < 1w EMA200
# Exit when KAMA reverses direction
# Uses trend-following on higher timeframe with weekly trend filter to avoid counter-trend trades
# Designed for 12h timeframe with selective entries to minimize trade frequency (target: 12-37/year)

name = "12h_KAMA_Trend_1wTrendFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate 1w EMA(200) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Calculate KAMA(10) on 12h
    # Efficiency ratio: |close - close[10]| / sum(|close - close[-1]|) over 10 periods
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Fix volatility calculation: rolling sum of absolute changes
    volatility = np.convolve(np.abs(np.diff(close, prepend=close[0])), np.ones(10), mode='same')
    volatility[:9] = np.nan  # First 9 values invalid
    
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9 (10th value)
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.diff(kama, prepend=np.nan)
    kama_dir = np.where(kama_dir > 0, 1, np.where(kama_dir < 0, -1, 0))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 10)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_dir[i]) or np.isnan(ema_200_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA trending up and price > 1w EMA200 (bullish filter)
            if kama_dir[i] == 1 and close[i] > ema_200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA trending down and price < 1w EMA200 (bearish filter)
            elif kama_dir[i] == -1 and close[i] < ema_200_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down
            if kama_dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up
            if kama_dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals