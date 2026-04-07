#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v1
Hypothesis: On 4-hour timeframe, use daily Camarilla pivot levels for mean reversion in ranging markets. 
Long when price touches S3 level with volume confirmation in ranging market (CHOP > 61.8).
Short when price touches R3 level with volume confirmation in ranging market.
Exit when price moves to opposite pivot level or volatility expands (CHOP < 38.2).
Designed for 20-40 trades/year to minimize fee drag while capturing mean reversion in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and CHOP
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3_1d = pivot_1d + (range_1d * 1.1 / 4)
    r2_1d = pivot_1d + (range_1d * 1.1 / 6)
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    s2_1d = pivot_1d - (range_1d * 1.1 / 6)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate Choppiness Index for 1d (CHOP)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - Wilder's smoothing
    period = 14
    alpha = 1.0 / period
    
    def wilders_smoothing(arr):
        smoothed = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                if i == 0:
                    smoothed[i] = np.nan
                else:
                    smoothed[i] = smoothed[i-1]
            else:
                if i == 0 or np.isnan(smoothed[i-1]):
                    smoothed[i] = arr[i]
                else:
                    smoothed[i] = smoothed[i-1] + alpha * (arr[i] - smoothed[i-1])
        return smoothed
    
    atr = wilders_smoothing(tr)
    
    # CHOP = 100 * log10(sum(ATR,14)/(n*(high-low))) / log10(n)
    # where n=14
    atr_sum = np.convolve(atr, np.ones(14), 'full')[:len(atr)]
    atr_sum[:13] = np.nan
    high_low_sum = np.convolve(range_1d, np.ones(14), 'full')[:len(range_1d)]
    high_low_sum[:13] = np.nan
    
    chop = 100 * np.log10(atr_sum / (14 * high_low_sum)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume ratio: current volume / 20-period average
    vol_ma = np.convolve(volume, np.ones(20)/20, 'same')
    vol_ma[:10] = np.nan
    vol_ma[-10:] = np.nan
    vol_ratio = volume / vol_ma
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
            
        # Range market condition: CHOP > 61.8
        ranging = chop_aligned[i] > 61.8
        trending = chop_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price reaches R1 or volatility expands (trending market)
            if close[i] >= r1_aligned[i] or trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S1 or volatility expands (trending market)
            if close[i] <= s1_aligned[i] or trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter in ranging market with volume confirmation
            if ranging and vol_ratio[i] > 1.5:
                # Long at S3
                if close[i] <= s3_aligned[i] and low[i] <= s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short at R3
                elif close[i] >= r3_aligned[i] and high[i] >= r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals