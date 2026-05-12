#!/usr/bin/env python3
"""
4h_12h_Camarilla_R3S3_Breakout_TrendVolume
Hypothesis: Breakouts at 12h Camarilla R3/S3 levels with 12h EMA50 trend filter and volume confirmation capture strong trends in both bull and bear markets while limiting trades to 20-50/year. 
Long when price breaks above R3 + price > 12h EMA50 + volume > 1.5x 20-period avg. 
Short when price breaks below S3 + price < 12h EMA50 + volume > 1.5x 20-period avg.
Exit when price returns to the opposite Camarilla level (S3 for longs, R3 for shorts) or trend reverses.
Designed for low trade frequency to minimize fee drag while capturing strong directional moves.
"""

name = "4h_12h_Camarilla_R3S3_Breakout_TrendVolume"
timeframe = "4h"
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

    # Get 12h data for Camarilla calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate 12h Camarilla levels (R3, S3)
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    hl_range_12h = high_12h - low_12h
    r3_12h = close_12h + hl_range_12h * 1.1 / 4
    s3_12h = close_12h - hl_range_12h * 1.1 / 4

    # Align Camarilla levels to 4h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + 12h uptrend + volume spike
            if close[i] > r3_12h_aligned[i-1] and close[i] > ema50_12h_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 12h downtrend + volume spike
            elif close[i] < s3_12h_aligned[i-1] and close[i] < ema50_12h_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to S3 or trend turns down
            if close[i] <= s3_12h_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to R3 or trend turns up
            if close[i] >= r3_12h_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals