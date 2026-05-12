#!/usr/bin/env python3
# 4h_RVI_Crossover_12hTrend_VolumeFilter
# Hypothesis: Relative Vigor Index (RVI) crossovers signal momentum shifts. Combined with 12h EMA trend filter and volume confirmation, this captures sustained moves in both bull and bear markets. RVI oscillates between -1 and 1, with crossovers above/below zero indicating bullish/bearish momentum. Trend filter ensures alignment with higher timeframe direction, reducing false signals. Volume filter confirms participation. Target: 20-40 trades/year.

name = "4h_RVI_Crossover_12hTrend_VolumeFilter"
timeframe = "4h"
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

    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')

    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # RVI (10-period) calculation
    # Numerator: close - open
    # Denominator: high - low
    num = close - prices['open'].values
    den = high - low
    # Avoid division by zero
    den = np.where(den == 0, 1e-10, den)
    # Smoothed numerator and denominator using EMA(10)
    num_smooth = pd.Series(num).ewm(span=10, adjust=False, min_periods=10).mean().values
    den_smooth = pd.Series(den).ewm(span=10, adjust=False, min_periods=10).mean().values
    rvi = num_smooth / den_smooth
    # Signal line: EMA of RVI
    rvi_signal = pd.Series(rvi).ewm(span=4, adjust=False, min_periods=4).mean().values

    # Volume filter: current > 1.3x average of last 20 bars
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after RVI warmup
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(rvi[i]) or 
            np.isnan(rvi_signal[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RVI crosses above signal line + 12h EMA50 uptrend + volume filter
            if (rvi[i] > rvi_signal[i] and rvi[i-1] <= rvi_signal[i-1] and
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RVI crosses below signal line + 12h EMA50 downtrend + volume filter
            elif (rvi[i] < rvi_signal[i] and rvi[i-1] >= rvi_signal[i-1] and
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RVI crosses below signal line (momentum fading)
            if rvi[i] < rvi_signal[i] and rvi[i-1] >= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RVI crosses above signal line (momentum fading)
            if rvi[i] > rvi_signal[i] and rvi[i-1] <= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals