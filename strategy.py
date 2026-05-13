#!/usr/bin/env python3
# 6h_Camarilla_Pivot_R3S3_Breakout_1dTrend_VolumeSpike_v2
# Hypothesis: Breakout beyond Camarilla R3/S3 levels on 6h timeframe with volume spike and aligned 1d trend.
# In uptrend (price > 1d EMA50), long breakout above R3; in downtrend (price < 1d EMA50), short breakdown below S3.
# Uses Camarilla levels from daily pivot (HLC of prior day) for institutional breakout levels.
# Volume surge confirms institutional participation. Trend filter avoids counter-trend breakouts.
# Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend).
# Low frequency due to strict breakout levels and volume confirmation.

name = "6h_Camarilla_Pivot_R3S3_Breakout_1dTrend_VolumeSpike_v2"
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

    # Get daily data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for prior day's pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior day's Camarilla levels (using prior day's HLC)
    # We shift by 1 to use prior day's data for current day's levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_val = high_1d - low_1d
    
    R3 = pivot + range_val * 1.1 / 2
    S3 = pivot - range_val * 1.1 / 2
    R4 = pivot + range_val * 1.1
    S4 = pivot - range_val * 1.1
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 4-period average (1 day worth at 6h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R3 + volume spike + daily uptrend
            if close[i] > R3_aligned[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 + volume spike + daily downtrend
            elif close[i] < S3_aligned[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below S3 (reversion to mean) OR trend reversal
            if close[i] < S3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above R3 (reversion to mean) OR trend reversal
            if close[i] > R3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals