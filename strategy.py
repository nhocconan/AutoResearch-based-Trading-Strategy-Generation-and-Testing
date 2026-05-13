#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_Volume_Confirmation
Hypothesis: Use daily Kaufman Adaptive Moving Average (KAMA) to capture trend direction, confirmed by volume spikes (volume > 1.5x 20-day average). KAMA adapts to market noise, reducing false signals in choppy markets while capturing true trends. Long when price crosses above KAMA with volume confirmation, short when price crosses below KAMA with volume confirmation. Designed for 1d timeframe to limit trades (<25/year) and avoid fee drag, effective in both bull (trend following) and bear (mean reversion during trend exhaustion) markets.
"""

name = "1d_KAMA_Trend_Filter_Volume_Confirmation"
timeframe = "1d"
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
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Kaufman Adaptive Moving Average (KAMA)
    # Parameters: fast=2, slow=30 (standard)
    fast_sc = 2 / (2 + 1)      # smoothing constant for fastest EMA
    slow_sc = 2 / (30 + 1)     # smoothing constant for slowest EMA
    
    # Calculate efficiency ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.where(volatility_sum != 0, change / volatility_sum, 0)
    
    # Calculate smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (no extra delay needed)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_aligned[i-1]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price crosses above KAMA with volume confirmation
            if close[i-1] <= kama_aligned[i-1] and close[i] > kama_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA with volume confirmation
            elif close[i-1] >= kama_aligned[i-1] and close[i] < kama_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals