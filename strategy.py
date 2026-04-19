#!/usr/bin/env python3
# 12h_TrueRangeBreakout_Volume_V2
# Hypothesis: 12h breakout of True Range with volume confirmation. Uses True Range to capture volatility breakouts, 
# works in both bull and bear by requiring volume confirmation to avoid false signals in low-volume chop.
# Targets 50-150 total trades over 4 years (12-37/year) by using strict volume threshold (2.0x average).

name = "12h_TrueRangeBreakout_Volume_V2"
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
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Average True Range (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Breakout levels: previous close ± 1.5 * ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # First period
    upper_break = prev_close + 1.5 * atr
    lower_break = prev_close - 1.5 * atr
    
    # Volume confirmation: volume > 2.0 * 20-period average (strict for fewer trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(atr[i]) or np.isnan(upper_break[i]) or np.isnan(lower_break[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper_break with volume confirmation
            if close[i] > upper_break[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower_break with volume confirmation
            elif close[i] < lower_break[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below previous close or volume dries up
            if close[i] < prev_close[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above previous close or volume dries up
            if close[i] > prev_close[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals