#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from 1d provide key support/resistance. Breakout above R1 or below S1 with volume confirmation and 1d EMA34 trend filter captures strong moves in both bull and bear markets. Target: 15-25 trades/year.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Calculate Camarilla levels from previous 1d bar
    # R1 = Close + (High - Low) * 1.12 / 12
    # S1 = Close - (High - Low) * 1.12 / 12
    # Using previous 1d bar to avoid look-ahead
    prev_1d_close = np.roll(close, 1)
    prev_1d_high = np.roll(high, 1)
    prev_1d_low = np.roll(low, 1)
    # First value will be invalid due to roll, handled by NaN check later
    camarilla_range = prev_1d_high - prev_1d_low
    r1 = prev_1d_close + camarilla_range * 1.12 / 12
    s1 = prev_1d_close - camarilla_range * 1.12 / 12

    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + volume spike + price above 1d EMA34 (bullish bias)
            if (close[i] > r1[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + volume spike + price below 1d EMA34 (bearish bias)
            elif (close[i] < s1[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend flip
            if close[i] < s1[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend flip
            if close[i] > r1[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals