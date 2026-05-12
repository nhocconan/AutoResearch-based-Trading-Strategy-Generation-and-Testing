#!/usr/bin/env python3
# 12h_1D_Camarilla_R3S3_Breakout_Volume
# Hypothesis: Breakouts above daily R3 or below daily S3 on 12h timeframe with volume confirmation. Uses daily timeframe for pivot levels to reduce noise and avoid overtrading. Designed for 12-37 trades/year to minimize fee drag while capturing strong momentum moves in both bull and bear markets. The 12h timeframe reduces trade frequency compared to 4h, lowering fee impact while still capturing significant moves.

name = "12h_1D_Camarilla_R3S3_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily Camarilla levels (R3 and S3) based on previous day
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4

    # Align daily levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above R3 with volume confirmation
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below S3 with volume confirmation
            elif close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3
            if close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals