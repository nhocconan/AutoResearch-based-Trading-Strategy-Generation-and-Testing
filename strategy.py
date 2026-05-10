#!/usr/bin/env python3
# 4h_1d_SMA_Breakout_Momentum
# Hypothesis: Breakouts above 200-day SMA with volume confirmation and trend alignment capture sustained momentum.
# The 200-day SMA acts as a major support/resistance level; breaks above/below indicate regime shifts.
# Volume surge validates breakout strength, reducing false signals. Works in bull/bear via directional filter.
# Targets 25-40 trades/year to minimize fee drag.

name = "4h_1d_SMA_Breakout_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for 200-day SMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily 200-day SMA
    close_1d = df_1d['close'].values
    sma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Align daily SMA to 4h timeframe
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # 4h data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need SMA200 (200) + volume MA (20)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(sma_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price position relative to 200-day SMA
        price_above_sma = close[i] > sma_200_1d_aligned[i]
        price_below_sma = close[i] < sma_200_1d_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above 200-day SMA with volume surge
            if price_above_sma and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 200-day SMA with volume surge
            elif price_below_sma and volume_surge:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below 200-day SMA
            if close[i] < sma_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above 200-day SMA
            if close[i] > sma_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals