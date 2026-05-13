#!/usr/bin/env python3
# 6h_Pivots_Fade_Trend_Filter
# Hypothesis: Fade at daily Camarilla R3/S3 with 1d EMA50 trend filter and volume confirmation.
# In uptrend (price > EMA50), short at R3 rejection; in downtrend (price < EMA50), long at S3 bounce.
# Uses mean reversion at extreme intraday levels with trend alignment to avoid counter-trend traps.
# Works in both bull (fade R3 in uptrend) and bear (fade S3 in downtrend) markets.
# Low frequency due to requirement of price reaching R3/S3 levels plus trend/volume filters.

name = "6h_Pivots_Fade_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    range_1d = high_1d - low_1d
    # Avoid division by zero
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    
    # Camarilla R3 and S3 levels
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 1.8 * 4-period average (1 day worth at 6h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 1.8 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at S3 bounce + downtrend + volume spike
            if low[i] <= s3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R3 rejection + uptrend + volume spike
            elif high[i] >= r3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above EMA50 or reaches midpoint
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if close[i] > ema50_1d_aligned[i] or close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below EMA50 or reaches midpoint
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if close[i] < ema50_1d_aligned[i] or close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals