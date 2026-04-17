#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Confirmation_v1
KAMA direction (adaptive moving average) with volume confirmation and 1-week EMA200 trend filter.
Long when KAMA rising and volume above average, price above weekly EMA200.
Short when KAMA falling and volume above average, price below weekly EMA200.
Exit when KAMA direction reverses.
Designed to capture sustained trends while avoiding choppy markets.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # will fix below
    # Recompute volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Vectorized volatility using convolution-like approach
    volatility = np.zeros(n)
    volatility[0] = 0
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Now compute ER for each point
    er = np.zeros(n)
    for i in range(10, n):
        if volatility[i] - volatility[i-10] > 0:
            er[i] = np.abs(close[i] - close[i-10]) / (volatility[i] - volatility[i-10])
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Volume average (20-period) ===
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # === 1-week EMA200 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # KAMA direction: rising if current > previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        # Volume confirmation: volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA rising, volume confirmed, price above weekly EMA200
            if (kama_rising and 
                vol_confirmed and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA falling, volume confirmed, price below weekly EMA200
            elif (kama_falling and 
                  vol_confirmed and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: KAMA direction reversal
        elif position == 1:
            # Exit long: KAMA starts falling
            if kama_falling:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA starts rising
            if kama_rising:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_Volume_Confirmation_v1"
timeframe = "1d"
leverage = 1.0