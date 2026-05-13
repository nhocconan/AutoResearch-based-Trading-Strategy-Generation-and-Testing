#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Fade at Camarilla R3/S3 levels with 1d trend filter and volume confirmation.
# In strong trends, price often reverses at Camarilla R3/S3 levels before continuing.
# We enter short at R3 in uptrends and long at S3 in downtrends, expecting a pullback.
# Volume confirms institutional interest at these key levels.
# 1d EMA50 filter ensures we trade with the higher timeframe trend.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Camarilla levels from previous day's range
    # Using previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 6
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 6
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)

    # Volume filter: >1.5x 20-period average on 6h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks below S3 (potential reversal) in downtrend + volume spike
            # We expect a bounce from S3 level in downtrend
            if (close[i] < S3_aligned[i] and 
                close[i] > S4_aligned[i] and  # Above S4 to avoid breakdown
                close[i] < ema_50_1d_aligned[i] and  # Downtrend filter
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks above R3 (potential reversal) in uptrend + volume spike
            # We expect a rejection from R3 level in uptrend
            elif (close[i] > R3_aligned[i] and 
                  close[i] < R4_aligned[i] and  # Below R4 to avoid breakout
                  close[i] > ema_50_1d_aligned[i] and  # Uptrend filter
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks above R3 (failure of bounce) or breaks below S4 (breakdown)
            if (close[i] > R3_aligned[i] or close[i] < S4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks below S3 (failure of rejection) or breaks above R4 (breakout)
            if (close[i] < S3_aligned[i] or close[i] > R4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals