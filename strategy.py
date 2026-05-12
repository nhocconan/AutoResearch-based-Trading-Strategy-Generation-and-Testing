#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: Use Camarilla pivot levels (R1/S1) from 1d combined with 1w EMA trend filter and volume confirmation (>2x 20-period average).
# Enter long when price breaks above R1 with bullish trend and volume spike; short when price breaks below S1 with bearish trend and volume spike.
# Exit when price returns to the 1d VWAP level or on trend reversal. Designed for 20-50 trades/year to minimize fee drag and work in both bull/bear markets via trend filter.

name = "4h_Camarilla_R1_S1_Breakout_Trend_Filter"
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

    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate Camarilla pivot levels for 1d
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rang = high_1d - low_1d
    r1_1d = close_1d + rang * 1.1 / 12
    s1_1d = close_1d - rang * 1.1 / 12
    # VWAP-like level: (H+L+C)/3
    vwap_1d = (high_1d + low_1d + close_1d) / 3

    # Align 1d levels to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)

    # 1w EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price > 1w EMA20 (bullish trend) + volume spike
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + price < 1w EMA20 (bearish trend) + volume spike
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to VWAP or trend turns bearish
            if (close[i] < vwap_1d_aligned[i] or 
                close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to VWAP or trend turns bullish
            if (close[i] > vwap_1d_aligned[i] or 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals