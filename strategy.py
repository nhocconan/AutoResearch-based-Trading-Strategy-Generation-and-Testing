#!/usr/bin/env python3
"""
1d_Weekly_KAMA_Trend_Signal_v1
Strategy: 1d trend using KAMA (adaptive moving average) with weekly trend filter.
Long when KAMA slopes upward and weekly close above weekly SMA50.
Short when KAMA slopes downward and weekly close below weekly SMA50.
Exit when trend reverses or price crosses opposite KAMA.
Uses weekly timeframe for trend filter to reduce whipsaw in ranging markets.
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    
    # Weekly SMA50 for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly SMA50 to daily timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate KAMA on daily data
    # Efficiency ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of absolute daily changes
    # Pad change array to match length
    change_padded = np.concatenate([np.full(10, np.nan), change])
    er = np.where(volatility != 0, change_padded / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (1-period change)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough for weekly SMA50 and KAMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close_1w[i // 7] > sma_50_1w[i // 7] if i >= 7 else False
        weekly_downtrend = close_1w[i // 7] < sma_50_1w[i // 7] if i >= 7 else False
        
        # KAMA direction
        kama_up = kama_slope[i] > 0
        kama_down = kama_slope[i] < 0
        
        if position == 0:
            # Long: weekly uptrend + KAMA rising
            if weekly_uptrend and kama_up:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + KAMA falling
            elif weekly_downtrend and kama_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly downtrend or KAMA falling
            if weekly_downtrend or kama_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly uptrend or KAMA rising
            if weekly_uptrend or kama_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_KAMA_Trend_Signal_v1"
timeframe = "1d"
leverage = 1.0