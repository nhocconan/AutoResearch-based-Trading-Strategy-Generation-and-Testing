#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use 4h close breaks of daily Camarilla R1/S1 levels with daily EMA50 trend filter and volume confirmation.
# This strategy targets 20-40 trades/year by requiring trend alignment and volume spikes, reducing false breakouts.
# Works in both bull and bear markets via daily trend filter that avoids counter-trend entries.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Camarilla levels from previous daily bar
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    prev_1d_close = df_1d['close'].shift(1).values

    # R1 = C + 1.1*(H-L)/6, S1 = C - 1.1*(H-L)/6
    r1 = prev_1d_close + 1.1 * (prev_1d_high - prev_1d_low) / 6
    s1 = prev_1d_close - 1.1 * (prev_1d_high - prev_1d_low) / 6

    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Volume confirmation: current volume > 2.0x average of last 4 periods (8 hours)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA50
        price_above_daily_ema = close[i] > ema_50_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Close breaks above R1 AND above daily EMA50 AND volume
            if close[i] > r1_aligned[i] and price_above_daily_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 AND below daily EMA50 AND volume
            elif close[i] < s1_aligned[i] and price_below_daily_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below R1 OR daily trend turns down
            if close[i] < r1_aligned[i] or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above S1 OR daily trend turns up
            if close[i] > s1_aligned[i] or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals