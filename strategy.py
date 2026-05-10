#!/usr/bin/env python3
"""
4H_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Price breaks Camarilla R3 or S3 levels with volume confirmation and 1d trend alignment.
Camarilla levels provide strong intraday support/resistance. Breakouts with volume and trend
filter capture momentum moves while avoiding false breakouts. Works in both bull and bear markets
by following the 1d trend direction. Target: 20-40 trades/year to minimize fee drag.
"""

name = "4H_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for the day
    # Camarilla formulas:
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # H2 = Close + 0.6 * (High - Low)
    # H1 = Close + 0.375 * (High - Low)
    # L1 = Close - 0.375 * (High - Low)
    # L2 = Close - 0.6 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # We use H3 (R3) and L3 (S3) as key levels
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * range_1d  # R3 level
    s3_1d = close_1d - 1.1 * range_1d  # S3 level
    
    # Align daily Camarilla levels to 4h timeframe (with 1-bar delay for completed daily bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 4h data for price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current 4h volume > 1.5x 20-period EMA
    volume_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > volume_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily data (2 days) and volume EMA (20)
    start_idx = max(2, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA34
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 + uptrend + volume
            if close[i] > r3_1d_aligned[i] and uptrend_1d and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + downtrend + volume
            elif close[i] < s3_1d_aligned[i] and downtrend_1d and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below EMA34 or breaks below S3 (mean reversion)
            if close[i] < ema34_1d_aligned[i] or close[i] < s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above EMA34 or breaks above R3 (mean reversion)
            if close[i] > ema34_1d_aligned[i] or close[i] > r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals