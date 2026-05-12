#!/usr/bin/env python3
"""
12h_Keltner_Channel_Breakout_1wTrend
Hypothesis: In trending markets, price breaks above/below the Keltner Channel (EMA20 ± 2*ATR) signal strong momentum. 
The 1-week EMA50 filter ensures we only trade in the direction of the higher timeframe trend, avoiding counter-trend whipsaws.
Volume confirmation (volume > 1.5x 20-period average) filters out low-momentum breakouts.
Works in both bull and bear markets by following the 1w trend direction.
"""

name = "12h_Keltner_Channel_Breakout_1wTrend"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate EMA20 and ATR(14) for Keltner Channel (12h timeframe)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Keltner Channel bounds
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr

    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Keltner Upper + 1w uptrend + volume spike
            if close[i] > kc_upper[i] and close[i] > ema50_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Keltner Lower + 1w downtrend + volume spike
            elif close[i] < kc_lower[i] and close[i] < ema50_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA20 or 1w trend turns down
            if close[i] < ema20[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA20 or 1w trend turns up
            if close[i] > ema20[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals