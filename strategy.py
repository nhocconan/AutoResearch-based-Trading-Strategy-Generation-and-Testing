#!/usr/bin/env python3
# 1d_KAMA_Trend_1wTrendFilter
# Strategy: Trade KAMA direction on 1d with 1w trend filter
# Long when KAMA direction up and price > 1w KAMA(30)
# Short when KAMA direction down and price < 1w KAMA(30)
# Exit when KAMA reverses
# Uses adaptive trend following with weekly filter to avoid counter-trend trades
# Designed for 1d timeframe with selective entries to minimize trade frequency

name = "1d_KAMA_Trend_1wTrendFilter"
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
    
    # Calculate 1w KAMA(30) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Efficiency Ratio (ER) for KAMA
    def calculate_kama(price, period=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        # Handle volatility = 0 case
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        # Initialize KAMA
        kama = np.full_like(price, np.nan)
        kama[period] = price[period]
        for i in range(period+1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama_30 = calculate_kama(close_1w, 30)
    kama_30_aligned = align_htf_to_ltf(prices, df_1w, kama_30)
    
    # Calculate KAMA on 1d for direction
    kama_1d = calculate_kama(close, 30)
    
    # Determine KAMA direction (1 = up, -1 = down, 0 = flat)
    kama_dir = np.zeros_like(kama_1d)
    kama_dir[30:] = np.where(kama_1d[30:] > kama_1d[29:-1], 1, 
                            np.where(kama_1d[30:] < kama_1d[29:-1], -1, 0))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 31  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_30_aligned[i]) or np.isnan(kama_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA direction up and price > 1w KAMA30 (uptrend filter)
            if kama_dir[i] == 1 and close[i] > kama_30_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA direction down and price < 1w KAMA30 (downtrend filter)
            elif kama_dir[i] == -1 and close[i] < kama_30_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA direction turns down
            if kama_dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA direction turns up
            if kama_dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals