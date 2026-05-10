#!/usr/bin/env python3
"""
12h_1d_KAMA_Direction_With_Volume_And_Chop_Filter
Hypothesis: Uses 1-day KAMA for trend direction, Choppiness Index for regime filtering,
and volume confirmation to enter trades aligned with the dominant trend on 12h timeframe.
Designed to avoid whipsaw in sideways markets and reduce trade frequency to avoid fee drag.
Long: Price crosses above 1d KAMA in uptrend (KAMA rising), chop < 61.8, volume > 1.5x average.
Short: Price crosses below 1d KAMA in downtrend (KAMA falling), chop < 61.8, volume > 1.5x average.
Exit: Trend reversal (price crosses KAMA in opposite direction) or chop > 61.8 (choppy market).
"""

name = "12h_1d_KAMA_Direction_With_Volume_And_Chop_Filter"
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
    
    # Get 1d data for KAMA trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (ER=10, fast=2, slow=30) on 1d close
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    vol = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])))
    if vol == 0:
        er = 0
    else:
        er = change / vol
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate Choppiness Index on 1d data (using high/low/close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_1d = np.zeros(len(close_1d))
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d[1:] = np.sum(np.atleast_2d(tr), axis=0) if len(tr) > 0 else 0  # Simplified for first value
    # Proper ATR calculation with smoothing
    atr_1d = pd.Series(atr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # True Range for current bar
    tr0 = np.maximum(
        np.abs(high_1d - low_1d),
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr0[0] = high_1d[0] - low_1d[0]  # First TR
    atr_smoothed = pd.Series(tr0).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr_smoothed).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Chop = 100 * log10(sum_atr / range_max_min) / log10(14)
    chop = 100 * np.log10(sum_atr / range_max_min) / np.log10(14)
    chop = np.where(range_max_min > 0, chop, 50)  # Avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5 * 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(close[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when market is not too choppy (Chop < 61.8)
        if chop_aligned[i] > 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA, KAMA rising (uptrend), volume confirmation
            if (close[i] > kama_aligned[i] and 
                close[i-1] <= kama_aligned[i-1] and 
                kama_aligned[i] > kama_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA, KAMA falling (downtrend), volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  close[i-1] >= kama_aligned[i-1] and 
                  kama_aligned[i] < kama_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: exit on trend reversal or chop regime shift
            if (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]) or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit on trend reversal or chop regime shift
            if (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]) or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals