#!/usr/bin/env python3
# 1d_1W_Camarilla_R1_S1_Breakout_Trend_VolumeS
# Hypothesis: Daily breakouts from weekly Camarilla R1/S1 levels in the direction of weekly trend (EMA50) with volume confirmation (1.5x 20-day average) to reduce false breakouts. Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag while capturing major trend moves in both bull and bear markets. Uses 1d timeframe with 1w HTF for trend filter.

name = "1d_1W_Camarilla_R1_S1_Breakout_Trend_VolumeS"
timeframe = "1d"
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

    # Get weekly data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly Camarilla levels (R1, S1) from previous week
    camarilla_range = high_1w - low_1w
    r1 = close_1w + 1.1 * camarilla_range / 12
    s1 = close_1w - 1.1 * camarilla_range / 12

    # Align Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)

    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate daily volume SMA20 for volume confirmation (spike filter)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R1 in weekly uptrend with volume spike confirmation
            if close[i] > r1_aligned[i] and close[i] > ema50_1w_aligned[i] and volume[i] > volume_spike_threshold[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 in weekly downtrend with volume spike confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema50_1w_aligned[i] and volume[i] > volume_spike_threshold[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly EMA50 (trend change)
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly EMA50 (trend change)
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals