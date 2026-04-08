#!/usr/bin/env python3
# 12h_1d_1w_camarilla_pivot_v2
# Hypothesis: 12-hour strategy using daily Camarilla pivot levels with weekly trend filter and volume confirmation.
# Long when: price touches S3 level and weekly close > weekly open (bullish week).
# Short when: price touches R3 level and weekly close < weekly open (bearish week).
# Exit when price moves to opposite H4/L4 level or weekly trend changes.
# Uses daily Camarilla for entry/exit levels and weekly trend filter to avoid counter-trend trades.
# Target: 12-30 trades/year to minimize fee drag while capturing meaningful reversals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_pivot_v2"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    n = len(high)
    pivot = (high + low + close) / 3
    range_val = high - low
    
    S1 = close - (range_val * 1.0 / 6)
    S2 = close - (range_val * 2.0 / 6)
    S3 = close - (range_val * 3.0 / 6)
    S4 = close - (range_val * 4.0 / 6)
    
    R1 = close + (range_val * 1.0 / 6)
    R2 = close + (range_val * 2.0 / 6)
    R3 = close + (range_val * 3.0 / 6)
    R4 = close + (range_val * 4.0 / 6)
    
    return S1, S2, S3, S4, R1, R2, R3, R4, pivot

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate average volume for confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    S1_1d, S2_1d, S3_1d, S4_1d, R1_1d, R2_1d, R3_1d, R4_1d, pivot_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Align Camarilla levels to 12h timeframe
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish if close > open, bearish if close < open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align weekly trend to 12h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        s3 = S3_1d_aligned[i]
        r3 = R3_1d_aligned[i]
        s4 = S4_1d_aligned[i]
        r4 = R4_1d_aligned[i]
        wb = weekly_bullish_aligned[i]
        wbear = weekly_bearish_aligned[i]
        
        if np.isnan(s3) or np.isnan(r3) or np.isnan(s4) or np.isnan(r4) or np.isnan(wb) or np.isnan(wbear):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        volume_ok = vol > avg_vol * 1.5  # Volume confirmation
        
        if position == 1:  # Long position
            # Exit: price reaches S4 level OR weekly trend turns bearish
            if price <= s4 or wbear > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R4 level OR weekly trend turns bullish
            if price >= r4 or wb > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions
            if volume_ok:
                # Long: price touches S3 level during bullish week
                if abs(price - s3) / s3 < 0.005 and wb > 0.5:  # Within 0.5% of S3
                    position = 1
                    signals[i] = 0.25
                # Short: price touches R3 level during bearish week
                elif abs(price - r3) / r3 < 0.005 and wbear > 0.5:  # Within 0.5% of R3
                    position = -1
                    signals[i] = -0.25
    
    return signals