#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 levels from 1d act as intraday support/resistance. Breakout with volume confirmation (>2x 20-period volume SMA) and 1d EMA50 trend filter ensures alignment with higher timeframe trend. Designed for low trade frequency (<25/year) to minimize fee drag in 12h timeframe. Works in bull/bear via trend filter and volatility-adjusted breakouts.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels (R1, S1) from previous 1d bar
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    rango_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rango_1d / 12
    camarilla_s1 = close_1d - 1.1 * rango_1d / 12

    # Align Camarilla levels to 12h timeframe (using previous 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 2x 20-period volume SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to access previous bar
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 with volume spike and uptrend
            if (close[i] > camarilla_r1_aligned[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 with volume spike and downtrend
            elif (close[i] < camarilla_s1_aligned[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below S1 (reversal signal)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above R1 (reversal signal)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals