#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance.
Breakout above R1 or below S1 with volume confirmation and daily trend filter.
Designed for 12-37 trades/year on 12h timeframe to work in both bull and bear markets.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Align to 12h timeframe (values from previous day's close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 12h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + uptrend + volume confirmation
            if close[i] > r1_val and close[i] > ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S1 + downtrend + volume confirmation
            elif close[i] < s1_val and close[i] < ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA34 or Camarilla S3 (strong reversal)
            camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4  # S3 level
            s3_aligned = align_htf_to_ltf(prices, df_1d, 
                                np.full_like(close_1d, camarilla_s3))
            if close[i] < ema34_val or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA34 or Camarilla R3 (strong reversal)
            camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4  # R3 level
            r3_aligned = align_htf_to_ltf(prices, df_1d, 
                                np.full_like(close_1d, camarilla_r3))
            if close[i] > ema34_val or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals