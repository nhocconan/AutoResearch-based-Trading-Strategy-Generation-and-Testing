#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_12hTrend_Volume_Confirmation
# Hypothesis: 4-hour breakouts above daily R1 or below daily S1 with volume confirmation and 12-hour trend filter.
# Uses daily timeframe for Camarilla pivot levels and 12-hour timeframe for trend direction to reduce noise and avoid overtrading.
# Daily pivot levels provide robust support/resistance levels, while 12h trend filter ensures trades align with higher timeframe momentum.
# Volume confirmation ensures institutional participation. Designed for 20-40 trades/year on 4H timeframe.

name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume_Confirmation"
timeframe = "4h"
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

    # Get daily data for Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # Calculate daily close for Camarilla calculation (use previous day)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate daily Camarilla levels (R1 and S1) based on previous day
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Handle first bar
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12

    # Align daily levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # 12h trend filter
        bullish_trend = close[i] > ema_12h_aligned[i]
        bearish_trend = close[i] < ema_12h_aligned[i]

        if position == 0:
            # LONG: Price closes above R1 with bullish 12h trend and volume confirmation
            if close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with bearish 12h trend and volume confirmation
            elif close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 or 12h trend turns bearish
            if close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 or 12h trend turns bullish
            if close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals