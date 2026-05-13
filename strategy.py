#!/usr/bin/env python3
# 12h_PivotPoint_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from daily Pivot Point R3/S3 levels on 12h timeframe with 1d EMA50 trend filter and volume spike confirmation.
# Uses standard pivot point calculation from previous day's OHLC (more stable than Camarilla in ranging markets).
# Trend filter: 1d EMA50 (only trade in direction of higher timeframe trend).
# Volume confirmation: current volume > 2.0 x 20-period average.
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation.
# Target: 15-35 trades/year per symbol to minimize fee drag.

name = "12h_PivotPoint_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate Pivot Points for 12h using previous day's OHLC
    # Standard pivot: P = (H + L + C) / 3
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    pivot_r3 = prev_high + 2 * (pivot - prev_low)
    pivot_s3 = prev_low - 2 * (prev_high - pivot)

    # Align Pivot levels to 12h timeframe
    pivot_r3_aligned = align_htf_to_ltf(prices, df_1d, pivot_r3)
    pivot_s3_aligned = align_htf_to_ltf(prices, df_1d, pivot_s3)

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(pivot_r3_aligned[i]) or 
            np.isnan(pivot_s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Pivot R3 in uptrend with volume spike
            if (close[i] > pivot_r3_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Pivot S3 in downtrend with volume spike
            elif (close[i] < pivot_s3_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Pivot S3 or trend turns down
            if close[i] < pivot_s3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Pivot R3 or trend turns up
            if close[i] > pivot_r3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals