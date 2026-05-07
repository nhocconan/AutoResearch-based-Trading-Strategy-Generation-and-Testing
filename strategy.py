#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filter_With_Volume_Confirmation
Hypothesis: Use KAMA direction on 4h as primary trend filter, enter long when price crosses above KAMA with volume confirmation (>1.5x average), short when price crosses below KAMA with volume confirmation. Uses daily timeframe for additional trend confirmation (price > daily EMA50 for longs, < for shorts) to avoid counter-trend trades. Designed for low trade frequency (~20-40/year) with high win rate in both bull and bear markets by requiring multiple confirmations.
"""

name = "4h_KAMA_Trend_Filter_With_Volume_Confirmation"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 4h
    # ER (Efficiency Ratio) = |change| / volatility
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Get daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for KAMA and indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(close_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price crosses above KAMA with uptrend and volume confirmation
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and  # crossover
                trend_up and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below KAMA with downtrend and volume confirmation
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and  # crossover
                  trend_down and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below KAMA or trend turns down
            if (close[i] < kama[i] and close[i-1] >= kama[i-1]) or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above KAMA or trend turns up
            if (close[i] > kama[i] and close[i-1] <= kama[i-1]) or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals