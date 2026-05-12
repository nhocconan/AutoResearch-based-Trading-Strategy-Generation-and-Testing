#!/usr/bin/env python3
"""
12h_Keltner_Breakout_Trend_Confirm
Hypothesis: Breakouts above/below Keltner Channel (20,2) with 1-day trend filter and volume confirmation on 12h timeframe.
Long: Close > upper Keltner + volume > 1.5x SMA20 + close > 1-day EMA50
Short: Close < lower Keltner + volume > 1.5x SMA20 + close < 1-day EMA50
Exit: Close crosses opposite Keltner band (lower for long, upper for short)
"""

name = "12h_Keltner_Breakout_Trend_Confirm"
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

    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Keltner Channel (20,2)
    atr_period = 20
    atr_multiplier = 2.0
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20 + atr_multiplier * atr
    lower_keltner = ema20 - atr_multiplier * atr

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 12h bar
        ema50_aligned = ema50_1d_aligned[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned) or np.isnan(vol_threshold_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper Keltner + volume spike + 1-day uptrend
            if (close[i] > upper_keltner[i] and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower Keltner + volume spike + 1-day downtrend
            elif (close[i] < lower_keltner[i] and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < lower Keltner
            if close[i] < lower_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close > upper Keltner
            if close[i] > upper_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals