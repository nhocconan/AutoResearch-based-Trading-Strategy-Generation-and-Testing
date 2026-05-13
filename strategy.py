#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d trend filter (EMA34) and volume confirmation. Works in bull via breakouts above R1 in uptrend and bear via breakdowns below S1 in downtrend. Uses 1d EMA34 as higher timeframe trend filter to avoid counter-trend trades. Target: 15-25 trades/year.

name = "12h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_Volume"
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

    # Calculate Camarilla pivot levels for 12h
    def calculate_camarilla(high, low, close):
        # Typical price
        pt = (high + low + close) / 3.0
        # Range
        r = high - low
        # Camarilla levels
        r1 = pt + (r * 1.1 / 12)
        s1 = pt - (r * 1.1 / 12)
        return r1, s1

    # We need previous bar's high/low/close for today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first value to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan

    r1, s1 = calculate_camarilla(prev_high, prev_low, prev_close)

    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume filter: >1.5x 24-period average (24*12h = 12 days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 (breakout) + price > EMA34 (uptrend) + volume spike
            if (close[i] > r1[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 (breakdown) + price < EMA34 (downtrend) + volume spike
            elif (close[i] < s1[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend turns down
            if (close[i] < r1[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend turns up
            if (close[i] > s1[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals