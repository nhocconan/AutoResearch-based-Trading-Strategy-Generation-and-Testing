#!/usr/bin/env python3
# 12h_RVI_Filtered_Camarilla_Breakout
# Hypothesis: Combines Relative Vigor Index (RVI) with Camarilla R1/S1 breakouts on 12h timeframe.
# Uses daily trend filter (EMA50) and volume confirmation (2.0x SMA20) to reduce false signals.
# RVI > 0.0 confirms bullish momentum for longs, RVI < 0.0 confirms bearish momentum for shorts.
# Designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull/bear markets.

name = "12h_RVI_Filtered_Camarilla_Breakout"
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
    open_price = prices['open'].values

    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate Camarilla levels from previous daily close
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.1 / 12
    s1 = close_1d - camarilla_range * 1.1 / 12

    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 2.0x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    # Relative Vigor Index (RVI) calculation
    # RVI = (Close - Open) / (High - Low) smoothed
    num = close - open_price
    den = high - low
    # Avoid division by zero
    den = np.where(den == 0, 1e-10, den)
    rvi_raw = num / den
    # Smooth with WMA (using SMA as proxy for efficiency)
    rvi = pd.Series(rvi_raw).rolling(window=10, min_periods=10).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 12h bar
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)[i]
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)[i]
        ema50_aligned = ema50_1d_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(r1_aligned) or np.isnan(s1_aligned) or 
            np.isnan(ema50_aligned) or np.isnan(volume_sma20[i]) or np.isnan(rvi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above Camarilla R1 + volume spike (2.0x) + daily uptrend + RVI > 0
            if (close[i] > r1_aligned and
                volume[i] > volume_threshold[i] and
                close[i] > ema50_aligned and
                rvi[i] > 0.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below Camarilla S1 + volume spike (2.0x) + daily downtrend + RVI < 0
            elif (close[i] < s1_aligned and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema50_aligned and
                  rvi[i] < 0.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S1 OR daily trend turns down OR RVI turns negative
            if close[i] < s1_aligned or close[i] < ema50_aligned or rvi[i] < 0.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R1 OR daily trend turns up OR RVI turns positive
            if close[i] > r1_aligned or close[i] > ema50_aligned or rvi[i] > 0.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals