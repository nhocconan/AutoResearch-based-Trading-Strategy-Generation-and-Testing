#!/usr/bin/env python3
# 12h_Keltner_Breakout_WeeklyTrend_Volume
# Hypothesis: On 12h timeframe, price breaking above/below Keltner Channel (2x ATR) with volume confirmation
# indicates momentum continuation. Weekly trend filter ensures we only trade in the direction of the
# higher timeframe trend, reducing whipsaw. Works in bull markets (breakouts continue up) and bear
# markets (breakouts continue down) by aligning with weekly trend. Volume filter ensures institutional
# participation, reducing false signals. Designed for 15-25 trades/year to minimize fee drag.

name = "12h_Keltner_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    # Extract data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # ATR(14) for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(close)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])

    # Keltner Channel (20-period EMA, 2*ATR)
    ema_close = np.zeros_like(close)
    ema_close[0] = close[0]
    for i in range(1, n):
        ema_close[i] = 0.1 * close[i] + 0.9 * ema_close[i-1]
    upper = ema_close + 2 * atr
    lower = ema_close - 2 * atr

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Weekly trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(atr[i]) or np.isnan(ema_close[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]

        if position == 0:
            # LONG: Price breaks above upper Keltner band with volume spike and weekly uptrend
            if close[i] > upper[i] and volume_spike[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner band with volume spike and weekly downtrend
            elif close[i] < lower[i] and volume_spike[i] and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA (trend change)
            if close[i] < ema_close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA (trend change)
            if close[i] > ema_close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals