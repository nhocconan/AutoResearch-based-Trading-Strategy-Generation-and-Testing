#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_12hTrend_VolumeS
# Hypothesis: 4h strategy using Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume confirmation.
# Designed for 19-50 trades/year per symbol. R1/S1 are tighter breakout levels than R3/S3, reducing trade frequency.
# Trend filter ensures trades align with medium-term trend, working in both bull and bear markets.
# Volume confirmation adds conviction to breakouts.

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeS"
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

    # Get 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)

    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Calculate Camarilla levels from previous 12h bar (R1/S1)
    # R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    prev_12h_high = df_12h['high'].shift(1).values
    prev_12h_low = df_12h['low'].shift(1).values
    prev_12h_close = df_12h['close'].shift(1).values

    r1 = prev_12h_close + 1.1 * (prev_12h_high - prev_12h_low) / 12
    s1 = prev_12h_close - 1.1 * (prev_12h_high - prev_12h_low) / 12

    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)

    # Volume confirmation: current volume > 1.5x average of last 4 periods (16 hours for 4h)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]

        if position == 0:
            # LONG: Close breaks above R1 AND uptrend AND volume
            if close[i] > r1_aligned[i] and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 AND downtrend AND volume
            elif close[i] < s1_aligned[i] and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below R1 OR trend turns down
            if close[i] < r1_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above S1 OR trend turns up
            if close[i] > s1_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals