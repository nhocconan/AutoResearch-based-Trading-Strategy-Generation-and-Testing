#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Filter
# Hypothesis: Breakout at 4h Camarilla R1/S1 with 4h trend filter and volume confirmation.
# Uses 1h for entry timing, 4h for direction (Camarilla levels, trend, volume).
# Targets 20-50 trades/year to avoid fee drag. Works in bull/bear via trend filter.
# Long: Close > 4h R1 + volume > 1.5x 4h SMA20 + close > 4h EMA50
# Short: Close < 4h S1 + volume > 1.5x 4h SMA20 + close < 4h EMA50
# Exit: Close crosses opposite 4h Camarilla level

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Filter"
timeframe = "1h"
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

    # Get 4h data for Camarilla levels, trend filter, and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values

    # Calculate Camarilla levels from previous 4h bar
    camarilla_range = high_4h - low_4h
    r1 = close_4h + camarilla_range * 1.1 / 12
    s1 = close_4h - camarilla_range * 1.1 / 12

    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # 4h volume SMA20 for confirmation
    volume_series_4h = pd.Series(volume_4h)
    volume_sma20_4h = volume_series_4h.rolling(window=20, min_periods=20).mean().values
    volume_threshold_4h = volume_sma20_4h * 1.5

    # Align 4h indicators to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_threshold_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_threshold_4h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_threshold_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > 4h R1 + volume > 1.5x SMA20 + close > 4h EMA50
            if (close[i] > r1_aligned[i] and
                volume[i] > volume_threshold_4h_aligned[i] and
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Close < 4h S1 + volume > 1.5x SMA20 + close < 4h EMA50
            elif (close[i] < s1_aligned[i] and
                  volume[i] > volume_threshold_4h_aligned[i] and
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < 4h S1
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close > 4h R1
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals