#!/usr/bin/env python3
# 6h_KAMA_Adaptive_Trend
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) with efficiency ratio to filter market noise.
# In trending markets (ER > 0.3), follows KAMA direction; in ranging markets (ER < 0.2), reverses at Bollinger Bands.
# Uses 1d EMA34 as higher timeframe trend filter for robustness. Designed for 6H timeframe with target 15-25 trades/year.

name = "6h_KAMA_Adaptive_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34  # EMA with alpha=2/(34+1)
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate KAMA components
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[i] - close[i-1]| over 10 periods
    
    # Pad arrays for alignment
    change_padded = np.full(n, np.nan)
    volatility_padded = np.full(n, np.nan)
    change_padded[10:] = change
    volatility_padded[10:] = volatility
    
    # Avoid division by zero
    er = np.full(n, np.nan)
    valid_vol = volatility_padded > 0
    er[valid_vol] = change_padded[valid_vol] / volatility_padded[valid_vol]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # for EMA2
    slow_sc = 2 / (30 + 1)  # for EMA30
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    if n > 0:
        kama[0] = close[0]
        for i in range(1, n):
            if not np.isnan(er[i]):
                sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
                kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # Bollinger Bands (20, 2) for mean reversion signals
    sma_20 = np.full(n, np.nan)
    std_20 = np.full(n, np.nan)
    
    if n >= 20:
        sma_20[19] = np.mean(close[0:20])
        std_20[19] = np.std(close[0:20])
        for i in range(20, n):
            sma_20[i] = np.mean(close[i-19:i+1])
            std_20[i] = np.std(close[i-19:i+1])
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime using Efficiency Ratio
        trending = er[i] > 0.3
        ranging = er[i] < 0.2
        
        if position == 0:
            # Enter long conditions
            long_signal = False
            if trending and close[i] > kama[i]:
                # In trend, go long when price above KAMA
                long_signal = True
            elif ranging and close[i] < lower_bb[i]:
                # In range, go long at lower Bollinger Band
                long_signal = True
            
            # Enter short conditions
            short_signal = False
            if trending and close[i] < kama[i]:
                # In trend, go short when price below KAMA
                short_signal = True
            elif ranging and close[i] > upper_bb[i]:
                # In range, go short at upper Bollinger Band
                short_signal = True
            
            # Apply 1d EMA34 trend filter: only take longs in uptrend, shorts in downtrend
            if long_signal and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif short_signal and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA OR 1d trend turns against position
            if close[i] < kama[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA OR 1d trend turns against position
            if close[i] > kama[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals