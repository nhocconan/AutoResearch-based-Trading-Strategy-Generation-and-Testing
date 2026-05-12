#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use 1d trend (close > EMA50) as filter, 4h Camarilla levels (R3/S3) for breakout direction, and 1h volume spike (>2x 20-period avg) for entry timing. Trades only during 08-20 UTC session. Targets 15-37 trades/year on 1h to avoid fee drag. Works in bull/bear via 1d trend filter.

name = "1h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "1h"
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

    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate previous 4h bar's Camarilla levels
    # Shift by 1 to use only completed 4h bar
    ph = np.roll(high_4h, 1)
    pl = np.roll(low_4h, 1)
    pc = np.roll(close_4h, 1)
    ph[0] = high_4h[0]
    pl[0] = low_4h[0]
    pc[0] = close_4h[0]

    # Camarilla formulas
    R3 = pc + (ph - pl) * 1.1 / 4
    S3 = pc - (ph - pl) * 1.1 / 4

    # Align to 1h
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to avoid roll issues
        # Skip if any required value is NaN or outside session
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_avg_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R3 (breakout) + price > 1d EMA50 + volume spike
            if (close[i] > R3_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.20
                position = 1
            # SHORT: Close < S3 (breakdown) + price < 1d EMA50 + volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < S3 (reversal below S3) or trend fails
            if close[i] < S3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close > R3 (reversal above R3) or trend fails
            if close[i] > R3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals