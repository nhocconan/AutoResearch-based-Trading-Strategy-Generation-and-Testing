#!/usr/bin/env python3
# 6h_1d_1w_camarilla_pivot_volume_v2
# Strategy: 6h Camarilla pivot reversals with weekly directional filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla levels (R3/S3) act as reversal zones in ranging markets. Weekly trend (1w) filters direction: 
# only take long reversals above weekly pivot, short reversals below. Volume > 2x 20-period average confirms 
# institutional interest at pivot touches. Designed for low trade frequency (~15-30/year) to minimize fee drag.
# Works in ranging markets via mean reversion at extremes and in trending via weekly filter alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_pivot_volume_v2"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical = (high + low + close) / 3
    range_ = high - low
    if range_ == 0:
        return typical, typical, typical, typical, typical, typical
    pivot = typical
    r3 = pivot + 1.1 * range_ / 2
    s3 = pivot - 1.1 * range_ / 2
    r4 = pivot + 1.1 * range_
    s4 = pivot - 1.1 * range_
    return pivot, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivots ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    typical_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    pivot_1d = typical_1d
    r3_1d = pivot_1d + 1.1 * range_1d / 2
    s3_1d = pivot_1d - 1.1 * range_1d / 2
    r4_1d = pivot_1d + 1.1 * range_1d
    s4_1d = pivot_1d - 1.1 * range_1d
    
    # Align daily Camarilla levels to 6h (no extra delay needed for pivot levels)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d.values)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d.values)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d.values)
    
    # Load weekly trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly pivot for trend filter (price above = bullish, below = bearish)
    typical_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pivot_1w = typical_1w
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w.values)
    
    # 6h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_avg_20[i]
        
        # Price touch conditions at Camarilla levels
        touch_r3 = abs(high[i] - r3_1d_aligned[i]) < 0.001 * r3_1d_aligned[i]  # Within 0.1%
        touch_s3 = abs(low[i] - s3_1d_aligned[i]) < 0.001 * s3_1d_aligned[i]  # Within 0.1%
        
        # Weekly trend filter
        weekly_bullish = close[i] > pivot_1w_aligned[i]
        weekly_bearish = close[i] < pivot_1w_aligned[i]
        
        # Entry conditions
        # Long: Touch S3 (support) AND weekly bullish AND volume confirmation
        if touch_s3 and weekly_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Touch R3 (resistance) AND weekly bearish AND volume confirmation
        elif touch_r3 and weekly_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite touch or weekly trend reversal
        elif position == 1 and (touch_r3 or not weekly_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (touch_s3 or not weekly_bearish):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals