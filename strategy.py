#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation
# Hypothesis: Camarilla pivot breakouts on 12h with daily trend filter and volume confirmation
# capture significant trend moves while avoiding whipsaw. Daily trend ensures alignment with
# higher-timeframe momentum, reducing false signals. Volume confirms breakout strength.
# Target: 12-30 trades/year (48-120 total over 4 years) to minimize fee drag.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate daily EMA20 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate Camarilla levels (R1, S1) on 12h data using previous day's range
    # We need previous day's high, low, close to calculate today's Camarilla levels
    # For 12h timeframe, we use 2-period lookback for previous day (since 2*12h = 24h)
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    for i in range(2, n):
        # Previous day's high, low, close (2 periods back for 12h timeframe)
        prev_high = high[i-2]
        prev_low = low[i-2]
        prev_close = close[i-2]
        # Camarilla R1 and S1 calculations
        R1[i] = prev_close + 1.1 * (prev_high - prev_low) / 12
        S1[i] = prev_close - 1.1 * (prev_high - prev_low) / 12

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 with volume spike and daily uptrend
            if close[i] > R1[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and daily downtrend
            elif close[i] < S1[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below S1 or daily trend turns down
            if close[i] < S1[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above R1 or daily trend turns up
            if close[i] > R1[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals