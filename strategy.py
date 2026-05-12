#!/usr/bin/env python3
# 1d_1w_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
# Hypothesis: Daily timeframe strategy using weekly Camarilla R1/S1 breakouts filtered by weekly trend (EMA20) and volume confirmation (1.5x average volume).
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws in ranging markets.
# Volume confirmation adds conviction to breakouts. Designed for low trade frequency (15-30 trades/year) to minimize fee drag.
# Works in bull markets by catching breakouts in uptrends and in bear markets by catching breakdowns in downtrends.

name = "1d_1w_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
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

    # Get weekly data for Camarilla pivot levels (R1, S1) and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values

    # Calculate weekly Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = close_1w + (high_1w - low_1w) * 1.1 / 12
    s1_1w = close_1w - (high_1w - low_1w) * 1.1 / 12

    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate weekly volume SMA10 for volume confirmation
    volume_series_1w = pd.Series(volume_1w)
    volume_sma10_1w = volume_series_1w.rolling(window=10, min_periods=10).mean().values
    volume_spike_threshold_1w = volume_sma10_1w * 1.5  # Require 1.5x average weekly volume

    # Align weekly indicators to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    volume_sma10_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma10_1w)
    volume_spike_threshold_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_threshold_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_spike_threshold_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above weekly R1 in weekly uptrend with volume spike
            if close[i] > r1_aligned[i] and close[i] > ema20_1w_aligned[i] and volume[i] > volume_spike_threshold_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S1 in weekly downtrend with volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema20_1w_aligned[i] and volume[i] > volume_spike_threshold_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals