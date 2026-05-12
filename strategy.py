#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h, buy when price touches or crosses above Camarilla R1 level with 1d uptrend (EMA34) and volume spike; sell when price touches or crosses below Camarilla S1 level with 1d downtrend and volume spike. Uses Camarilla levels from daily timeframe for institutional-grade support/resistance, combined with trend filter and volume confirmation to avoid false breakouts. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing institutional order flow in both bull and bear markets.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+C)/3 (typical price), but commonly simplified to close
    # Using standard formula: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    camarilla_r1 = close_1d + camarilla_range
    camarilla_s1 = close_1d - camarilla_range

    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 24-period average (2 days of 12h data)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches or crosses above Camarilla R1 + 1d uptrend + volume spike
            if (close[i] >= camarilla_r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or crosses below Camarilla S1 + 1d downtrend + volume spike
            elif (close[i] <= camarilla_s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches or crosses below Camarilla S1 OR trend turns down
            if close[i] <= camarilla_s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches or crosses above Camarilla R1 OR trend turns up
            if close[i] >= camarilla_r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals