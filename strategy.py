#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1wTrend_Volume
# Hypothesis: Price breaking out of Camarilla R1/S1 levels on 12h with 1w trend filter and volume confirmation captures strong momentum moves while avoiding whipsaws. Works in both bull and bear markets by using weekly trend filter to avoid counter-trend trades.

name = "12h_Camarilla_R1S1_Breakout_1wTrend_Volume"
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

    # Calculate ATR for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # 1w EMA34 for trend filter (load once, align)
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: volume > 2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Calculate Camarilla levels for current 12h bar (using previous bar's range)
        if i > 0:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            
            if range_val > 0:
                camarilla_multiplier = 1.1 / 12
                r1 = prev_close + range_val * camarilla_multiplier * 1
                s1 = prev_close - range_val * camarilla_multiplier * 1
            else:
                r1 = prev_close
                s1 = prev_close
        else:
            r1 = close[0]
            s1 = close[0]

        if position == 0:
            # LONG: Close above R1 + 1w EMA34 uptrend + volume spike
            if (close[i] > r1 and 
                close[i] > ema34_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S1 + 1w EMA34 downtrend + volume spike
            elif (close[i] < s1 and 
                  close[i] < ema34_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below 1w EMA34 or volatility drop
            if close[i] < ema34_1w_aligned[i] or volume[i] < vol_avg_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above 1w EMA34 or volatility drop
            if close[i] > ema34_1w_aligned[i] or volume[i] < vol_avg_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals