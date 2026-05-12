#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
# Hypothesis: Camarilla R1/S1 levels from 4H combined with 12H EMA trend filter and volume spike.
# Camarilla levels provide precise support/resistance for breakouts, while 12H EMA ensures alignment with higher timeframe trend.
# Volume spike confirms breakout strength. Designed for moderate trade frequency (~30-50/year) to balance signal quality and fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    c = close
    h = high
    l = low
    R4 = c + (h - l) * 1.1 / 2
    R3 = c + (h - l) * 1.1 / 4
    R2 = c + (h - l) * 1.1 / 6
    R1 = c + (h - l) * 1.1 / 12
    S1 = c - (h - l) * 1.1 / 12
    S2 = c - (h - l) * 1.1 / 6
    S3 = c - (h - l) * 1.1 / 4
    S4 = c - (h - l) * 1.1 / 2
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    close_12h = df_12h['close'].values

    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate Camarilla levels from previous 4h bar
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    
    for i in range(1, n):
        R1, R2, R3, R4, S1, S2, S3, S4 = calculate_camarilla(high[i-1], low[i-1], close[i-1])
        camarilla_R1[i] = R1
        camarilla_S1[i] = S1

    # Volume spike: 2.0x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 with volume spike and uptrend (close > 12h EMA50)
            if (close[i] > camarilla_R1[i] and
                volume[i] > volume_spike_threshold[i] and
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 with volume spike and downtrend (close < 12h EMA50)
            elif (close[i] < camarilla_S1[i] and
                  volume[i] > volume_spike_threshold[i] and
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below S1 (reversal signal)
            if close[i] < camarilla_S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above R1 (reversal signal)
            if close[i] > camarilla_R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals