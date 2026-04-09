#!/usr/bin/env python3
# 4h_camarilla_1d_trend_volume_v5
# Hypothesis: 4h strategy using Camarilla pivot levels from 1d for mean reversion in ranging markets,
# filtered by 12h choppiness regime to avoid trending markets. Volume confirmation on entry.
# Long: price touches S3 + CHOP > 61.8 (range) + volume spike
# Short: price touches R3 + CHOP > 61.8 (range) + volume spike
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_1d_trend_volume_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Pivot = (high + low + close) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    # Range = high - low
    rang = high_1d - low_1d
    # Resistance levels
    r4 = close_1d + rang * 1.500
    r3 = close_1d + rang * 1.250
    r2 = close_1d + rang * 1.166
    r1 = close_1d + rang * 1.083
    # Support levels
    s1 = close_1d - rang * 1.083
    s2 = close_1d - rang * 1.166
    s3 = close_1d - rang * 1.250
    s4 = close_1d - rang * 1.500
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h HTF data for choppiness regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Choppiness Index: CHOP = 100 * log10(sum(atr1) / (n * log(n+1))) / log10(n)
    # Simplified: CHOP = 100 * log10(atr_sum / (true_range_max * n)) / log10(n)
    # Where atr_sum = sum of true range over n periods
    # true_range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = np.maximum(high_12h - low_12h, np.abs(high_12h - np.roll(close_12h, 1)))
    tr1 = np.maximum(tr1, np.abs(low_12h - np.roll(close_12h, 1)))
    tr1[0] = high_12h[0] - low_12h[0]  # first period
    
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    tr_max = pd.Series(tr1).rolling(window=14, min_periods=14).max().values
    
    # Avoid division by zero
    chop_raw = np.where(tr_max > 0, atr_sum / (tr_max * 14), 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price moves above S2 (mean reversion complete) OR chop < 38.2 (trending)
            if close[i] > s2_aligned[i] if not np.isnan(s2_aligned[i]) else False or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves below R2 (mean reversion complete) OR chop < 38.2 (trending)
            if close[i] < r2_aligned[i] if not np.isnan(r2_aligned[i]) else False or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need chop > 61.8 (ranging market) and volume confirmation
            if chop_aligned[i] > 61.8 and volume[i] > 2.0 * volume_ma[i]:
                # Long: price touches S3 (strong support)
                if close[i] <= s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches R3 (strong resistance)
                elif close[i] >= r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals