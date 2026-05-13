#!/usr/bin/env python3
# 1d_KeltnerChannel_Breakout_Volume_Trend
# Hypothesis: Keltner Channel (20,2) breakouts capture strong trends. 
# Use weekly EMA200 as trend filter to avoid counter-trend trades. 
# Volume confirmation (2x 20-period average) filters weak breakouts.
# Works in bull via upside breaks, bear via downside breaks. 
# Target: 15-25 trades/year.

name = "1d_KeltnerChannel_Breakout_Volume_Trend"
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

    # Keltner Channel: EMA(20) +/- 2 * ATR(10)
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).values
    kc_upper = ema20 + 2 * atr10
    kc_lower = ema20 - 2 * atr10

    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Volume filter: >2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema20[i]) or np.isnan(atr10[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > Keltner Upper + weekly uptrend + volume spike
            if (close[i] > kc_upper[i] and 
                close[i] > ema200_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < Keltner Lower + weekly downtrend + volume spike
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema200_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < EMA20 or trend reversal
            if close[i] < ema20[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close > EMA20 or trend reversal
            if close[i] > ema20[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals