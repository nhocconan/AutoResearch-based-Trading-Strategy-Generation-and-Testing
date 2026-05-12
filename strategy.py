#!/usr/bin/env python3
# 4h_Keltner_Breakout_Trend_Volume_Switch
# Hypothesis: Keltner Channel breakout with 1d trend filter and volume confirmation. 
# Long: Close > upper band + volume > 1.5x avg volume + close > 1d EMA200
# Short: Close < lower band + volume > 1.5x avg volume + close < 1d EMA200
# Exit: Close crosses middle line (EMA20)
# Uses 4h timeframe for balance of signal frequency and noise reduction.
# Works in bull via trend-following breaks, in bear via mean-reversion at bands.

name = "4h_Keltner_Breakout_Trend_Volume_Switch"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate EMA20 and ATR for Keltner Channels (4h)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.maximum(high - low, 
                               np.maximum(np.abs(high - np.roll(close, 1)), 
                                          np.abs(low - np.roll(close, 1))))).rolling(
        window=10, min_periods=10).mean().values
    atr[0] = 0

    upper_band = ema20 + 2 * atr
    lower_band = ema20 - 2 * atr
    middle_band = ema20  # EMA20

    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # Volume confirmation: 1.5x 20-period average volume
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_ma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 4h bar
        ema200_aligned = ema200_1d_aligned[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(ema200_aligned) or np.isnan(vol_threshold_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper band + volume spike + daily uptrend
            if (close[i] > upper_band[i] and
                volume[i] > vol_threshold_val and
                close[i] > ema200_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower band + volume spike + daily downtrend
            elif (close[i] < lower_band[i] and
                  volume[i] > vol_threshold_val and
                  close[i] < ema200_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below middle band (EMA20)
            if close[i] < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above middle band (EMA20)
            if close[i] > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals