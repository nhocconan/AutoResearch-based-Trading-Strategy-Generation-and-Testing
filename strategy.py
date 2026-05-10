#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyFilter
Hypothesis: Price breaks Camarilla R3 (long) or S3 (short) from prior day's range, with 1d EMA34 trend filter and weekly trend confirmation.
This strategy uses stronger breakout levels (R3/S3) for fewer, higher-quality trades and adds weekly trend filter to avoid counter-trend moves in bear markets.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyFilter"
timeframe = "6h"
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
    
    # 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels from prior day: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # 1d EMA34 for trend filter
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    
    # Align 1d indicators to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Align weekly indicators to 6h
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters
        is_uptrend = close[i] > ema34_1d_aligned[i] and close[i] > ema34_1w_aligned[i]
        is_downtrend = close[i] < ema34_1d_aligned[i] and close[i] < ema34_1w_aligned[i]
        price_above_r3 = close[i] > r3_aligned[i]
        price_below_s3 = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3, in uptrend (both 1d and 1w)
            if price_above_r3 and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, in downtrend (both 1d and 1w)
            elif price_below_s3 and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R3 or trend turns down (either timeframe)
            if not price_above_r3 or not (close[i] > ema34_1d_aligned[i] and close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S3 or trend turns up (either timeframe)
            if not price_below_s3 or not (close[i] < ema34_1d_aligned[i] and close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals