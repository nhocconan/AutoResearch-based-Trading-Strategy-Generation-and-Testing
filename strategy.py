#!/usr/bin/env python3
# 1D_Camarilla_Pivot_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Price breaking above R3 or below S3 Camarilla levels on daily chart with weekly trend filter and volume confirmation captures strong momentum. Weekly trend ensures we trade with the dominant higher timeframe direction, reducing whipsaw in sideways markets. Volume spike confirms institutional participation. Works in bull/bear via weekly trend filter.

name = "1D_Camarilla_Pivot_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Calculate Camarilla levels from previous day
    # R4 = Close + ((High - Low) * 1.1 / 2)
    # R3 = Close + ((High - Low) * 1.1 / 4)
    # S3 = Close - ((High - Low) * 1.1 / 4)
    # S4 = Close - ((High - Low) * 1.1 / 2)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]

    rng = prev_high - prev_low
    r3 = prev_close + (rng * 1.1 / 4)
    s3 = prev_close - (rng * 1.1 / 4)

    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume filter: >1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R3 + weekly uptrend + volume spike
            if (close[i] > r3[i] and 
                close[i] > ema34_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 + weekly downtrend + volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema34_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S3 or trend reversal
            if close[i] < s3[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R3 or trend reversal
            if close[i] > r3[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals