#!/usr/bin/env python3
"""
6h_Keltner_Channel_Reversal
Hypothesis: Price reversals at Keltner Channel bands (2*ATR) with 1d trend filter and volume confirmation capture mean-reversion in ranging markets and pullbacks in trends. Designed for 50-150 total trades over 4 years to minimize fee drag while working in both bull and bear regimes.
"""

name = "6h_Keltner_Channel_Reversal"
timeframe = "6h"
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

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values

    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Calculate ATR(14) for Keltner Channels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Calculate Keltner Channels (2*ATR)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr

    # Align 1d EMA34 to 6h timeframe with 1-day delay (need previous day's close)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(ema20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at lower Keltner + 1d uptrend + volume above average
            vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
            if close[i] <= lower_keltner[i] and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg * 1.2:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at upper Keltner + 1d downtrend + volume above average
            elif close[i] >= upper_keltner[i] and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg * 1.2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above EMA20 or 1d trend turns down
            if close[i] >= ema20[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below EMA20 or 1d trend turns up
            if close[i] <= ema20[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals