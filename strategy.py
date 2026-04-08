#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v1
# Hypothesis: Use 1d Camarilla pivot levels (H3/L3) on 12h timeframe with volume confirmation and choppiness regime filter.
# Long: Price breaks above H3 with volume > 1.5x average AND chop > 61.8 (trending regime)
# Short: Price breaks below L3 with volume > 1.5x average AND chop > 61.8 (trending regime)
# Exit: Opposite pivot break or chop < 38.2 (range regime) to avoid false breakouts
# Uses 12h primary timeframe with 1d HTF for Camarilla levels and chop filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate average volume for confirmation (20-period SMA)
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot levels and choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # H3 = Pivot + 1.1 * (H - L) / 2
    # L3 = Pivot - 1.1 * (H - L) / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    h3_1d = pivot_1d + 1.1 * (high_1d - low_1d) / 2.0
    l3_1d = pivot_1d - 1.1 * (high_1d - low_1d) / 2.0
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR1) / (n * ATRn)) / log10(n)
    # Where ATR1 = true range, ATRn = n-period ATR
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    # Handle first bar
    tr1[0] = high_1d[0] - low_1d[0]
    
    atr1 = tr1
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR1 over 14 periods
    sum_atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index
    chop_1d = 100 * np.log10(sum_atr1 / (14 * atr14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop_1d = np.where((atr14 > 0) & (sum_atr1 > 0), chop_1d, 50.0)
    
    # Align 1d indicators to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price breaks below L3 (opposite pivot break)
            # 2. Chop < 38.2 (range regime) - avoid false breakouts in ranging markets
            if low[i] < l3_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price breaks above H3 (opposite pivot break)
            # 2. Chop < 38.2 (range regime)
            if high[i] > h3_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H3 with volume confirmation AND chop > 61.8 (trending regime)
            if (high[i] > h3_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                chop_1d_aligned[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 with volume confirmation AND chop > 61.8 (trending regime)
            elif (low[i] < l3_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  chop_1d_aligned[i] > 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals