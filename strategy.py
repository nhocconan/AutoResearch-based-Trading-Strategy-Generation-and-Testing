#!/usr/bin/env python3
"""
12h_Price_Channel_Breakout_Volume
Hypothesis: Price channel breakouts (Donchian/ATR-based) combined with volume spikes and trend filters on 12h timeframe work in both bull and bear markets by capturing institutional moves. Uses 1d trend filter (EMA34) and volume confirmation to reduce false signals. Target: 15-30 trades/year per symbol to minimize fee drag.
"""
name = "12h_Price_Channel_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # ATR(14) for stop loss and channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.zeros_like(tr)
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing

    # Donchian Channel (20-period)
    upper = np.zeros_like(high)
    lower = np.zeros_like(low)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Load daily EMA34 for trend filter (updated only after daily close)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian + volume spike + above daily EMA34
            if close[i] > upper[i] and volume_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + volume spike + below daily EMA34
            elif close[i] < lower[i] and volume_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below lower Donchian or ATR-based trailing stop
            if close[i] < lower[i] or close[i] < (np.maximum.accumulate(high)[i] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above upper Donchian or ATR-based trailing stop
            if close[i] > upper[i] or close[i] > (np.minimum.accumulate(low)[i] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals