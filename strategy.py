#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: On 1h timeframe, buy when price breaks above 4h Camarilla R1 with 4h uptrend (EMA34) and 1d volume spike; sell when price breaks below 4h Camarilla S1 with 4h downtrend and 1d volume spike. Uses 4h for signal direction/trend, 1d for volume confirmation, and 1h only for entry timing. Session filter (08-20 UTC) reduces noise. Targets 15-35 trades/year to avoid fee drag.
# Uses discrete position sizing (0.20) to minimize churn and respects all MTF data loading rules.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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

    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate Camarilla levels from previous 4h bar
    range_4h = high_4h - low_4h
    camarilla_r1 = close_4h + range_4h * 1.12 / 12
    camarilla_s1 = close_4h - range_4h * 1.12 / 12

    # Use previous 4h bar's levels (shift by 1)
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan

    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_prev)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_prev)

    # 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)

    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values

    # 1d volume average (20-period ~20 days)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN or outside session
        if (not in_session[i] or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 4h Camarilla R1 + 4h uptrend + 1d volume spike
            if (close[i] > camarilla_r1_aligned[i] and
                close[i] > ema34_4h_aligned[i] and
                volume_1d[-1] > vol_avg_20_aligned[i] * 2.0):  # Use latest 1d volume (updated daily)
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 4h Camarilla S1 + 4h downtrend + 1d volume spike
            elif (close[i] < camarilla_s1_aligned[i] and
                  close[i] < ema34_4h_aligned[i] and
                  volume_1d[-1] > vol_avg_20_aligned[i] * 2.0):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 4h Camarilla S1 OR trend turns down
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 4h Camarilla R1 OR trend turns up
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals