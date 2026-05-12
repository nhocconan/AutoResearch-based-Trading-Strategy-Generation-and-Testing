#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_Volume_Limited
Hypothesis: Breakout above R1 or below S1 from daily Camarilla pivots with 12h EMA200 trend filter and volume confirmation. 
Uses longer-term EMA200 for stronger trend filtering to reduce trades and improve robustness across bull/bear markets.
Designed for 20-30 trades/year on 4h timeframe.
"""

name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume_Limited"
timeframe = "4h"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    hl_range = high_1d - low_1d
    r1_1d = close_1d + hl_range * 1.1 / 12
    s1_1d = close_1d - hl_range * 1.1 / 12

    # Align to 4h timeframe (values from previous day's close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)

    # Get 12h data for trend filter (EMA200 for stronger filtering)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)

    # Volume confirmation: 4h volume > 2x 30-period average (stricter)
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        ema200_val = ema200_12h_aligned[i]
        vol_avg_val = vol_avg_30[i]

        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema200_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + uptrend (above EMA200) + volume confirmation
            if close[i] > r1_val and close[i] > ema200_val and volume[i] > vol_avg_val * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S1 + downtrend (below EMA200) + volume confirmation
            elif close[i] < s1_val and close[i] < ema200_val and volume[i] > vol_avg_val * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA200 or below S1 (re-entry level)
            if close[i] < ema200_val or close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA200 or above R1 (re-entry level)
            if close[i] > ema200_val or close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals