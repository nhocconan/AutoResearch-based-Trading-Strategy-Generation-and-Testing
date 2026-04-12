#!/usr/bin/env python3
"""
4h_1d_ema100_ema200_vol_trend_v1
Hypothesis: Use 1-day EMA100 and EMA200 crossovers on the 4h chart to determine trend direction, with volume confirmation for entry.
Long when EMA100 > EMA200 (uptrend) and price crosses above EMA200 with volume.
Short when EMA100 < EMA200 (downtrend) and price crosses below EMA100 with volume.
Exit when trend reverses or price crosses back to the opposite EMA.
Designed to work in both bull and bear markets by following the daily trend and using volume to filter false signals.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

name = "4h_1d_ema100_ema200_vol_trend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA100 and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA100 and EMA200 for trend direction
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA100 and EMA200 to 4h timeframe
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: EMA100 > EMA200 (uptrend) AND price crosses above EMA200 with volume
        if (ema100_1d_aligned[i] > ema200_1d_aligned[i] and 
            close[i] > ema200_1d_aligned[i] and 
            close[i-1] <= ema200_1d_aligned[i-1] and 
            vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: EMA100 < EMA200 (downtrend) AND price crosses below EMA100 with volume
        elif (ema100_1d_aligned[i] < ema200_1d_aligned[i] and 
              close[i] < ema100_1d_aligned[i] and 
              close[i-1] >= ema100_1d_aligned[i-1] and 
              vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or price crosses back to opposite EMA
        elif position == 1 and (ema100_1d_aligned[i] < ema200_1d_aligned[i] or close[i] < ema100_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (ema100_1d_aligned[i] > ema200_1d_aligned[i] or close[i] > ema200_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals