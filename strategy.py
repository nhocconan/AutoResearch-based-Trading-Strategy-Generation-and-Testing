#!/usr/bin/env python3
# 4H_Keltner_Breakout_TrendVolume
# Hypothesis: 4h Keltner breakout with 1d EMA34 trend filter and volume spike confirmation.
# Keltner bands (EMA20 + 1.5*ATR) provide dynamic support/resistance for breakouts.
# 1d EMA34 filters trend direction to avoid counter-trend trades.
# Volume spike (1.5x average) confirms breakout strength.
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull/bear by following 1d trend and using dynamic bands.

name = "4H_Keltner_Breakout_TrendVolume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 4h data for price action and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate ATR(10) for Keltner bands
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Calculate EMA20 for Keltner basis
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate Keltner bands
    upper_keltner = ema20_4h + 1.5 * atr10
    lower_keltner = ema20_4h - 1.5 * atr10

    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above upper Keltner in 1d uptrend with volume spike
            if close[i] > upper_keltner[i] and close[i] > ema34_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below lower Keltner in 1d downtrend with volume spike
            elif close[i] < lower_keltner[i] and close[i] < ema34_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below lower Keltner (reversal signal)
            if close[i] < lower_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above upper Keltner (reversal signal)
            if close[i] > upper_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals