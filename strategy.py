#!/usr/bin/env python3
"""
4h_KAMA_Direction_Volume_Trend_Filter
Hypothesis: Price follows KAMA (adaptive trend) direction with volume confirmation and EMA trend filter.
Uses KAMA to capture trend direction in both bull and bear markets, with volume surge to confirm momentum.
Designed to avoid whipsaws by requiring strong volume on trend entries. Target: 20-25 trades/year (80-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Adaptive Moving Average) - more responsive in trends, stable in ranges
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # temporary fix, will replace below
    
    # Proper volatility calculation for ER
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # EMA20 trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for EMA20 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_20[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        ema20 = ema_20[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above both KAMA and EMA20 with volume confirmation
            if price > kama_val and price > ema20 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below both KAMA and EMA20 with volume confirmation
            elif price < kama_val and price < ema20 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA or EMA20
            if price < kama_val or price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA or EMA20
            if price > kama_val or price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0