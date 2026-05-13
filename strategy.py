#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Enter long when price breaks above Camarilla R3 level during 1d uptrend with volume spike.
# Enter short when price breaks below Camarilla S3 level during 1d downtrend with volume spike.
# Camarilla levels from daily pivot provide institutional support/resistance.
# Trend filter ensures alignment with higher timeframe momentum.
# Volume surge confirms institutional participation.
# Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend).
# Low frequency due to strict breakout and volume confirmation requirements.

name = "4h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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

    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)  # R3 = pivot + 1.1 * range / 2
    s3 = pivot - (range_1d * 1.1 / 2)  # S3 = pivot - 1.1 * range / 2
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 6-period average (1 day worth at 4h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6
    
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
            # LONG: Close > R3 + daily uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S3 + daily downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA50 OR below S3 (stop loss)
            if close[i] < ema50_1d_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA50 OR above R3 (stop loss)
            if close[i] > ema50_1d_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals