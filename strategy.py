#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_VolumeFilter
Hypothesis: On 6h timeframe, buy when price breaks above weekly pivot S3/R3 with volume >2x average and weekly trend up; sell when price breaks below weekly pivot S3/R3 with volume >2x average and weekly trend down. Uses weekly pivot levels as dynamic support/resistance with volume confirmation and trend filter to capture strong breakouts while minimizing false signals. Designed to work in both bull and bear markets by following the weekly trend. Targets 15-30 trades per year to minimize fee drag.
"""

name = "6h_WeeklyPivot_Breakout_VolumeFilter"
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

    # Get weekly data for pivot points and trend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Calculate S3 and R3 levels
    weekly_s3 = weekly_low - 2.0 * weekly_range
    weekly_r3 = weekly_high + 2.0 * weekly_range
    
    # Weekly trend using EMA20
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Align weekly data to 6h timeframe (wait for weekly close)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_w, weekly_s3)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_w, weekly_r3)
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_w, weekly_ema20)

    # Volume confirmation: volume > 2x 24-period average (approx 4 days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_s3_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly R3 + weekly uptrend + volume spike
            if (close[i] > weekly_r3_aligned[i] and 
                close[i] > weekly_ema20_aligned[i] and 
                volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S3 + weekly downtrend + volume spike
            elif (close[i] < weekly_s3_aligned[i] and 
                  close[i] < weekly_ema20_aligned[i] and 
                  volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly S3 OR trend turns down
            if close[i] < weekly_s3_aligned[i] or close[i] < weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly R3 OR trend turns up
            if close[i] > weekly_r3_aligned[i] or close[i] > weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals