#!/usr/bin/env python3
"""
12h_Pivot_Range_Breakout_1wTrend_VolumeSpike
Hypothesis: Price breaking above/below weekly pivot range (based on weekly high-low-close) with 1-week EMA trend filter and volume confirmation (2x average) captures strong trending moves while avoiding false breakouts. Works in bull/bear by following 1-week trend direction. Uses 12h timeframe to minimize trade frequency and maximize statistical significance.
"""

name = "12h_Pivot_Range_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
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

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly pivot range (based on previous week's high-low-close)
    # Pivot range: S1 = 2*P - H, R1 = 2*P - L where P = (H+L+C)/3
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Shift by 1 to use previous week's data
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan

    pivot_point = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    weekly_support = 2 * pivot_point - prev_high_1w  # S1
    weekly_resistance = 2 * pivot_point - prev_low_1w  # R1

    # Align weekly pivot levels to 12h timeframe
    weekly_support_aligned = align_htf_to_ltf(prices, df_1w, weekly_support)
    weekly_resistance_aligned = align_htf_to_ltf(prices, df_1w, weekly_resistance)

    # 1-week EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume spike: >2x 50-period average (12h)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(weekly_support_aligned[i]) or np.isnan(weekly_resistance_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly resistance + 1w EMA50 uptrend + volume spike
            if (close[i] > weekly_resistance_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly support + 1w EMA50 downtrend + volume spike
            elif (close[i] < weekly_support_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly support (reversal level)
            if close[i] < weekly_support_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly resistance (reversal level)
            if close[i] > weekly_resistance_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals