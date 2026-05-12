#!/usr/bin/env python3

# 12h_1D_Camarilla_R1S1_Breakout_Trend_Volume
# Hypothesis: Breakout above daily R1 with 12h price > 12h EMA21 (trend) and volume confirmation.
# In bear markets, short below daily S1 with 12h price < 12h EMA21 and volume confirmation.
# Uses 12h timeframe for low frequency (12-37 trades/year target) and daily Camarilla levels for structure.
# Weekly trend filter is omitted to reduce complexity; trend is derived from 12h EMA21.
# Volume confirmation requires current volume > 1.5x 20-period average to avoid false breakouts.

name = "12h_1D_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate 12h EMA21 for trend filter
    ema_12h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values

    # Calculate daily Camarilla levels (based on previous day)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12

    # Align daily levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(21, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above R1 with 12h uptrend and volume confirmation
            if close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] and close[i] > ema_12h[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below S1 with 12h downtrend and volume confirmation
            elif close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] and close[i] < ema_12h[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or 12h trend turns down
            if close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] or close[i] < ema_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or 12h trend turns up
            if close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] or close[i] > ema_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals