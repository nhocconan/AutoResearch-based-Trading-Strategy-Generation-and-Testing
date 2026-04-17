#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d EMA50 Trend Filter
Long: Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50
Short: Bear Power < 0 AND Bull Power > 0 AND price < 1d EMA50
Exit: Opposite power signal or price crosses 1d EMA50
Uses Elder Ray (Bull/Bear power) to measure bull/bear strength relative to EMA13,
filtered by 1d EMA50 trend for higher timeframe bias.
Designed to work in trending markets with clear bull/bear separation.
Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Get 1d data for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(13, 50)  # need EMA13 and 1d EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50
            if bull_val > 0 and bear_val < 0 and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND price < 1d EMA50
            elif bear_val < 0 and bull_val > 0 and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Opposite power signal or price < 1d EMA50
            if bull_val <= 0 or bear_val >= 0 or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Opposite power signal or price > 1d EMA50
            if bull_val >= 0 or bear_val <= 0 or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0