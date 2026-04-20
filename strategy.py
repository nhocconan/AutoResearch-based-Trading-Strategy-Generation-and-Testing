#!/usr/bin/env python3
# 12h_KAMA_Trend_Filter_Strategy
# Hypothesis: On 12h timeframe, use Kaufman Adaptive Moving Average (KAMA) to determine trend direction.
# Enter long when price crosses above KAMA with volume confirmation and low chop regime.
# Enter short when price crosses below KAMA with volume confirmation and low chop regime.
# Use chop filter (Choppiness Index) to avoid ranging markets and reduce false signals.
# Position size 0.25 to manage risk and reduce drawdown. Target 12-37 trades/year.

name = "12h_KAMA_Trend_Filter_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for chop filter (Choppiness Index)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate KAMA (10, 2, 30) - ER=10, Fast=2, Slow=30
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    abs_sum = np.zeros_like(close)
    for i in range(1, len(close)):
        abs_sum[i] = abs_sum[i-1] + np.abs(close[i] - close[i-1])
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.maximum(abs_sum[10:], 1e-10)  # Avoid division by zero
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Chop filter (Choppiness Index) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of True Range over 14 periods
    tr_sum_14 = np.full_like(tr_1d, np.nan)
    for i in range(len(tr_1d)):
        if i >= 13:  # 14-period sum
            tr_sum_14[i] = np.nansum(tr_1d[i-13:i+1])
    
    # High-Low range over 14 periods
    max_high_14 = np.full_like(high_1d, np.nan)
    min_low_14 = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 13:
            max_high_14[i] = np.max(high_1d[i-13:i+1])
            min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop = np.full_like(close_1d, 50.0)  # Default to middle
    for i in range(len(chop)):
        if not np.isnan(tr_sum_14[i]) and tr_sum_14[i] > 0:
            range_14 = max_high_14[i] - min_low_14[i]
            if range_14 > 0:
                chop[i] = 100 * np.log10(tr_sum_14[i] / range_14) / np.log10(14)
    
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure KAMA and volume MA are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when chop < 61.8 (trending market)
        if chop_aligned[i] >= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume confirmation
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume confirmation
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals