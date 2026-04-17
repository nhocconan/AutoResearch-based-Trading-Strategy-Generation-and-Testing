#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Filter
Strategy: 1d KAMA direction with volume confirmation.
Long: KAMA rising + volume > 1.5x 20-period average
Short: KAMA falling + volume > 1.5x 20-period average
Exit: KAMA direction reversal
Position size: 0.25
Designed to capture trending moves while avoiding chop.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    kama_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Net change over kama_period
    net_change = np.abs(np.subtract(close[kama_period:], close[:-kama_period]))
    net_change = np.concatenate([np.full(kama_period, np.nan), net_change])
    
    # Sum of absolute changes over kama_period
    sum_abs_change = np.convolve(abs_change, np.ones(kama_period), mode='full')[:len(close)]
    sum_abs_change = np.concatenate([np.full(kama_period-1, np.nan), sum_abs_change[kama_period-1:]])
    
    # Efficiency Ratio
    er = np.where(sum_abs_change != 0, net_change / sum_abs_change, 0)
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Get 1w trend (close > open = uptrend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    trend_1w = (df_1w['close'] > df_1w['open']).astype(float).values  # 1 for up, 0 for down
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get volume average (20-period)
    volume_ma20 = np.convolve(volume, np.ones(20)/20, mode='full')[:len(volume)]
    volume_ma20 = np.concatenate([np.full(19, np.nan), volume_ma20[19:]])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from sufficient warmup
    start_idx = max(kama_period*2, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(trend_1w_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Current volume
        volume_current = volume[i]
        volume_filter = volume_current > (1.5 * volume_ma20[i])
        
        # KAMA direction
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Entry signals
        if position == 0:
            # Long: KAMA rising + volume filter + 1w uptrend
            if kama_rising and volume_filter and trend_1w_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + volume filter + 1w downtrend
            elif kama_falling and volume_filter and trend_1w_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling
            if not kama_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising
            if not kama_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0