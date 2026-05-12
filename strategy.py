#!/usr/bin/env python3
"""
12h_Keltner_Breakout_Trend_Volume
Hypothesis: Use Keltner Channel breakouts on 12h with weekly trend filter (EMA34) and volume confirmation to capture major trend moves. Works in bull markets via upward breakouts and in bear markets via downward breakouts, with the weekly trend filter reducing counter-trend trades. Target: 15-30 trades/year.
"""

name = "12h_Keltner_Breakout_Trend_Volume"
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

    # Get weekly data for trend filter (call once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    close_weekly = df_weekly['close'].values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)

    # Keltner Channel on 12h (period=20, multiplier=2.0)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        ema20_val = ema20[i]
        upper_val = upper[i]
        lower_val = lower[i]
        ema34_val = ema34_weekly_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(ema20_val) or np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above upper Keltner + weekly uptrend + volume confirmation
            if close[i] > upper_val and close[i] > ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below lower Keltner + weekly downtrend + volume confirmation
            elif close[i] < lower_val and close[i] < ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below middle line (EMA20) or weekly trend turns down
            if close[i] < ema20_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above middle line (EMA20) or weekly trend turns up
            if close[i] > ema20_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals