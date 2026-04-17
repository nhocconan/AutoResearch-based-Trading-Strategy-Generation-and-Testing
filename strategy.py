#!/usr/bin/env python3
"""
6h_Stochastic_Signal_v1
Stochastic %K(14,3,3) > 80 for short, < 20 for long with 1d trend filter.
Trend filter: price above/below 1d EMA50.
Exit when Stochastic crosses back to neutral zone (40-60).
Designed to capture mean reversion in ranging markets with trend alignment.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Stochastic Oscillator (14,3,3) ===
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    # Smooth %K to get %D (3-period SMA of %K)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(d_percent[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Stochastic < 20 (oversold) and price above 1d EMA50 (uptrend)
            if (d_percent[i] < 20 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Stochastic > 80 (overbought) and price below 1d EMA50 (downtrend)
            elif (d_percent[i] > 80 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Stochastic > 40 (exit oversold zone)
            if d_percent[i] > 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Stochastic < 60 (exit overbought zone)
            if d_percent[i] < 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Stochastic_Signal_v1"
timeframe = "6h"
leverage = 1.0