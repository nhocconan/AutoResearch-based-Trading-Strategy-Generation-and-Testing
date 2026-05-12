#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1wTrend_Confirm
# Hypothesis: Camarilla R3/S3 breakouts filtered by weekly trend (price above/below weekly EMA50) with volume confirmation.
# The weekly trend filter ensures alignment with dominant market direction, reducing whipsaws in counter-trend moves.
# Works in both bull and bear by following weekly trend. Target: 15-30 trades/year.

name = "6h_Camarilla_R3S3_Breakout_1wTrend_Confirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Previous day's Camarilla levels
    # R3 = C + 1.1*(H-L)*1.1/2 = C + 1.1*(H-L)*0.55
    # S3 = C - 1.1*(H-L)*1.1/2 = C - 1.1*(H-L)*0.55
    camarilla_shift = 1.1 * (high_1d - low_1d) * 0.55
    r3_1d = close_1d + camarilla_shift
    s3_1d = close_1d - camarilla_shift

    # Align Camarilla levels to 6h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)

    # Volume confirmation: current > 1.5x average of last 24 periods (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after weekly EMA50 warmup
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + weekly uptrend + volume confirmation
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + weekly downtrend + volume confirmation
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below previous day's close (mean reversion)
            if close[i] < close_1d[i // 4]:  # 4x 6h bars = 1 day
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above previous day's close (mean reversion)
            if close[i] > close_1d[i // 4]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals