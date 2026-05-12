#!/usr/bin/env python3
"""
4h_1d_Camarilla_R3S3_Breakout_Trend_Volume
Hypothesis: On 4h timeframe, buy when price breaks above the 1-day Camarilla R3 level with volume >1.8x average and 1-day EMA34 trending up; sell when price breaks below the 1-day S3 level with volume >1.8x average and 1-day EMA34 trending down. Uses 1-day Camarilla pivot levels with volume confirmation and trend filter to capture strong trends while minimizing false breakouts. Targets 20-50 trades per year to reduce fee drift.
"""

name = "4h_1d_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1-day Camarilla levels (based on previous day's range)
    # Camarilla formulas: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous day's data to avoid look-ahead
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan

    # Calculate Camarilla levels
    camarilla_r3 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 2
    camarilla_s3 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 2

    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.8x 20-period average (approx 10 hours)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R3 + 1d uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + 1d downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR trend turns down
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR trend turns up
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals