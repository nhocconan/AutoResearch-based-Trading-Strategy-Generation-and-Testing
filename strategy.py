#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter
# Hypothesis: Daily Camarilla pivot levels (R1/S1) breakouts with weekly trend filter and volume confirmation
# capture trending moves with low whipsaw. Works in bull (breakout above R1 in uptrend) and bear
# (breakdown below S1 in downtrend) markets. Weekly trend ensures alignment with higher-timeframe momentum.
# Volume filter confirms breakout strength. Target: 20-30 trades/year.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly EMA20 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate previous day's Camarilla pivot levels (R1, S1)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    for i in range(1, n):
        camarilla_r1[i] = close[i-1] + 1.1 * (high[i-1] - low[i-1]) / 12.0
        camarilla_s1[i] = close[i-1] - 1.1 * (high[i-1] - low[i-1]) / 12.0

    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 with volume spike and weekly uptrend
            if close[i] > camarilla_r1[i] and volume_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 with volume spike and weekly downtrend
            elif close[i] < camarilla_s1[i] and volume_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla pivot point (CP) or weekly trend turns down
            camarilla_pivot = (high[i-1] + low[i-1] + close[i-1]) / 3.0
            if close[i] < camarilla_pivot or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla pivot point (CP) or weekly trend turns up
            camarilla_pivot = (high[i-1] + low[i-1] + close[i-1]) / 3.0
            if close[i] > camarilla_pivot or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals