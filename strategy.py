#!/usr/bin/env python3
# 4h_12h_Camarilla_R1S1_Breakout_Volume_Trend
# Hypothesis: Camarilla R1/S1 levels derived from daily price action provide strong intraday support/resistance.
# Enter long when price breaks above R1 with volume spike and 12h EMA50 uptrend; short when breaks below S1 with volume spike and downtrend.
# Exit on opposite Camarilla level touch or trend reversal. Designed to work in both bull and bear markets by using 12h trend filter.
# Uses 4h for entry timing and 12h for trend direction, minimizing false signals while capturing meaningful moves.

name = "4h_12h_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "4h"
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

    # Get 12h data for trend filter and Camarilla calculation (using daily for Camarilla)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 2:
        return np.zeros(n)

    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate Camarilla levels from daily data
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12

    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Calculate 20-period SMA of volume for volume spike detection
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1, volume spike, and 12h trend up
            if (close[i] > camarilla_r1_aligned[i] and
                volume[i] > 2.0 * vol_sma20[i] and
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1, volume spike, and 12h trend down
            elif (close[i] < camarilla_s1_aligned[i] and
                  volume[i] > 2.0 * vol_sma20[i] and
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches S1 or trend turns down
            if (close[i] < camarilla_s1_aligned[i] or
                ema50_12h_aligned[i] < ema50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches R1 or trend turns up
            if (close[i] > camarilla_r1_aligned[i] or
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals