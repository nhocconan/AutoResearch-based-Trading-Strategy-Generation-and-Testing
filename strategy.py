#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12-hour KAMA trend direction and volume spike confirmation.
Uses KAMA (Kaufman Adaptive Moving Average) to detect adaptive trend on 12h timeframe.
Enters long when 12h KAMA slope is positive and volume > 1.5x average volume.
Enters short when 12h KAMA slope is negative and volume > 1.5x average volume.
Exits when KAMA slope changes sign or volume drops below average.
Target: 20-60 total trades over 4 years (5-15/year) with size 0.25 to minimize fee drag.
"""

name = "4h_KAMA_Trend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12-hour KAMA (Kaufman Adaptive Moving Average)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close']
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = abs(close_12h.diff(er_length))
    volatility = close_12h.diff().abs().rolling(window=er_length, min_periods=er_length).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close_12h))
    kama[0] = close_12h.iloc[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_12h.iloc[i] - kama[i-1])
    
    # Calculate KAMA slope (1-period change)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # Align KAMA slope to 4h timeframe
    kama_slope_aligned = align_htf_to_ltf(prices, df_12h, kama_slope)
    
    # Volume spike detection: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_slope_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: positive KAMA slope + volume spike
            if kama_slope_aligned[i] > 0 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: negative KAMA slope + volume spike
            elif kama_slope_aligned[i] < 0 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA slope turns negative OR volume drops below average
            if kama_slope_aligned[i] <= 0 or volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA slope turns positive OR volume drops below average
            if kama_slope_aligned[i] >= 0 or volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals