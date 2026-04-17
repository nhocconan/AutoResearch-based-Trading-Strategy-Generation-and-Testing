#!/usr/bin/env python3
"""
12h_1d_TRIX_VolumeRegime
Strategy: 12-hour TRIX momentum with volume spike and 1d chop regime filter.
Long: TRIX crosses above zero + volume > 1.5x 20-period avg + 1d chop > 61.8 (range)
Short: TRIX crosses below zero + volume > 1.5x 20-period avg + 1d chop > 61.8 (range)
Exit: TRIX crosses back to zero
Position size: 0.25
Designed to capture momentum bursts in ranging markets with volume confirmation.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX (12-period EMA applied 3 times)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1-day data for chop regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day chopiness index (14-period)
    atr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    # Pad ATR array to match length
    atr_1d = np.concatenate([np.array([np.nan]), atr_1d])
    
    # True range for chop calculation
    tr_1d = atr_1d
    atr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max and min over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop index: 100 * log10(ATR_sum / (max_high - min_low)) / log10(14)
    range_1d = max_high - min_low
    chop_1d = np.where(
        (range_1d > 0) & (~np.isnan(range_1d)) & (~np.isnan(atr_sum)),
        100 * np.log10(atr_sum / range_1d) / np.log10(14),
        50  # default to neutral when invalid
    )
    
    # Align 1d chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # need chop and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Chop filter: chop > 61.8 indicates ranging market
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # TRIX zero cross
        trix_cross_up = (i > 0) and (trix[i-1] <= 0) and (trix[i] > 0)
        trix_cross_down = (i > 0) and (trix[i-1] >= 0) and (trix[i] < 0)
        
        if position == 0:
            # Long: TRIX crosses up + volume filter + chop filter
            if trix_cross_up and volume_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses down + volume filter + chop filter
            elif trix_cross_down and volume_filter and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses back down
            if trix_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses back up
            if trix_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_TRIX_VolumeRegime"
timeframe = "12h"
leverage = 1.0