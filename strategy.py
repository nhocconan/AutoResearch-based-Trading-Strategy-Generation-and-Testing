#!/usr/bin/env python3
# 12h_Weekly_Camarilla_R3S3_Breakout_Trend_Filter
# Hypothesis: Use weekly timeframe to calculate Camarilla R3/S3 levels from the previous week.
# Breakouts above R3 (long) or below S3 (short) are taken only when aligned with the 1-week EMA50 trend.
# Volume confirmation (current volume > 1.5x 4-period average) adds conviction.
# Designed for 12-37 trades/year per symbol, works in both bull and bear via trend filter.

name = "12h_Weekly_Camarilla_R3S3_Breakout_Trend_Filter"
timeframe = "12h"
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

    # Get weekly data for trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # 1-week EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Calculate Camarilla levels from previous weekly bar
    prev_1w_high = df_1w['high'].shift(1).values
    prev_1w_low = df_1w['low'].shift(1).values
    prev_1w_close = df_1w['close'].shift(1).values

    # R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    r3 = prev_1w_close + 1.1 * (prev_1w_high - prev_1w_low) / 2
    s3 = prev_1w_close - 1.1 * (prev_1w_high - prev_1w_low) / 2

    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)

    # Volume confirmation: current volume > 1.5x average of last 4 periods (48 hours)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1-week EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]

        if position == 0:
            # LONG: Close breaks above R3 AND uptrend AND volume
            if close[i] > r3_aligned[i] and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3 AND downtrend AND volume
            elif close[i] < s3_aligned[i] and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below R3 OR trend turns down
            if close[i] < r3_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above S3 OR trend turns up
            if close[i] > s3_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals