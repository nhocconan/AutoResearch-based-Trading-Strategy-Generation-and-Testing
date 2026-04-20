#!/usr/bin/env python3
# 4h_1d_KAMA_Trend_Filtered_Breakout
# Hypothesis: On 4h timeframe, trade breakouts from 4h price channels when aligned with 1d KAMA trend direction.
# Uses 1d KAMA to filter trades in trending markets and 4h Donchian breakout for entry with volume confirmation.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Targets 20-30 trades per year to minimize fee drag.

name = "4h_1d_KAMA_Trend_Filtered_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA (ER=10)
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    
    # Efficiency Ratio
    change = abs(close_1d_series - close_1d_series.shift(10))
    volatility = abs(close_1d_series - close_1d_series.shift(1)).rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Seed
    
    for i in range(10, len(close_1d)):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_1d[i] - kama[i-1])
    
    # 4h Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(kama_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, volume spike, and price above 1d KAMA (uptrend)
            if (close[i] > donch_high[i] and 
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > kama_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, volume spike, and price below 1d KAMA (downtrend)
            elif (close[i] < donch_low[i] and 
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < kama_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reversal (below KAMA)
            if close[i] < donch_low[i] or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reversal (above KAMA)
            if close[i] > donch_high[i] or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals