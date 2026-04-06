#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w KAMA trend filter and volume confirmation
# Long when price breaks above Donchian high + KAMA up + volume > 1.5x average
# Short when price breaks below Donchian low + KAMA down + volume > 1.5x average
# Exit when price crosses KAMA or Donchian midpoint reverses
# Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Works in trending markets by following breakouts with trend filter
# 1w trend filter ensures we only trade with the major trend, reducing whipsaws

name = "12h_donchian_kama_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 12h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # KAMA (Kaufman Adaptive Moving Average) - 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Efficiency Ratio
    er = np.zeros_like(close_1w)
    for i in range(10, len(close_1w)):  # 10-period ER
        direction = np.abs(close_1w[i] - close_1w[i-9])
        volatility = np.sum(np.abs(np.diff(close_1w[i-9:i+1])))
        er[i] = direction / (volatility + 1e-10)
    
    # Smoothing constants (fast=2, slow=30)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(kama_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses KAMA or Donchian midpoint
        if position == 1:  # long position
            if close[i] <= kama_aligned[i] or close[i] <= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= kama_aligned[i] or close[i] >= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            # Bullish breakout: price above Donchian high + KAMA up + volume
            if (close[i] > donch_high[i] and 
                kama_aligned[i] > kama_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + KAMA down + volume
            elif (close[i] < donch_low[i] and 
                  kama_aligned[i] < kama_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals