#!/usr/bin/env python3
# 12h_1d_volatility_breakout_with_volume
# Hypothesis: 12-hour volatility breakout using 1-day ATR with volume confirmation
# Works in bull/bear by capturing volatility expansion after consolidation, with volume filter to avoid false breakouts.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "12h_1d_volatility_breakout_with_volume"
timeframe = "12h"
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
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR (14-day ATR)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR multiplier for breakout threshold (0.5 * ATR)
    atr_threshold = atr_14d * 0.5
    
    # Calculate 20-period high/low for breakout levels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align ATR threshold and breakout levels to 12h timeframe
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1d, atr_threshold)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_threshold_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above 20-period high + volatility expansion + volume
        if (close[i] > high_20_aligned[i] and 
            (high[i] - low[i]) > atr_threshold_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below 20-period low + volatility expansion + volume
        elif (close[i] < low_20_aligned[i] and 
              (high[i] - low[i]) > atr_threshold_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal
        elif position == 1 and close[i] < low_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > high_20_aligned[i]:
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