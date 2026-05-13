#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume
# Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike captures momentum with controlled frequency. Works in bull/bear via 12h trend filter and volatility-based levels.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
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

    # ATR for risk context (not used in signal)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Camarilla levels from previous day
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Handle first bar
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    range_val = prev_high - prev_low
    camarilla_r1 = prev_close + range_val * 1.1 / 12
    camarilla_s1 = prev_close - range_val * 1.1 / 12

    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume filter: >2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + 12h EMA50 uptrend + volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema50_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S1 + 12h EMA50 downtrend + volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or volume drop
            if close[i] < camarilla_s1[i] or volume[i] < vol_avg_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or volume drop
            if close[i] > camarilla_r1[i] or volume[i] < vol_avg_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals