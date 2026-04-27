#!/usr/bin/env python3
"""
4h_Pivot_Mean_Reversion_with_Volume_and_Trend_Filter
Hypothesis: Mean reversion at daily pivot support/resistance levels during low volatility regimes. 
Goes long when price touches S1 in uptrend (price > 200 EMA) with volume spike, short when touches R1 in downtrend (price < 200 EMA) with volume spike.
Uses 1d pivot levels, 4h EMA200 trend filter, and volume > 2x 20-period average. 
Designed for low-frequency, high-probability trades (target: 20-40/year) that work in both bull and bear markets.
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
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate daily pivot points: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    s1 = 2 * pivot - high_1d    # Support 1
    r1 = 2 * pivot - low_1d     # Resistance 1
    
    # Align pivot levels to 4h timeframe (previous day's levels available at open)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume filter: require volume > 2x 20-period average to avoid low-volatility false signals
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches S1 support in uptrend (price > EMA200) with volume spike
            if (close[i] <= s1_aligned[i] * 1.002 and  # allow small slippage
                close[i] > ema200_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 resistance in downtrend (price < EMA200) with volume spike
            elif (close[i] >= r1_aligned[i] * 0.998 and  # allow small slippage
                  close[i] < ema200_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches pivot point or trend fails
            if (close[i] >= pivot[i] or 
                close[i] <= ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches pivot point or trend fails
            if (close[i] <= pivot[i] or 
                close[i] >= ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_Mean_Reversion_with_Volume_and_Trend_Filter"
timeframe = "4h"
leverage = 1.0