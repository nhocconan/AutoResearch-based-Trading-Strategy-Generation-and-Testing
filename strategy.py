#!/usr/bin/env python3

# 4h_1D_Camarilla_R1S1_Breakout_Trend_Volume
# Hypothesis: Buy near Camarilla R1 in uptrend, sell near S1 in downtrend on 4h timeframe with daily trend filter and volume confirmation.
# Uses 1d trend filter (EMA34) to align with higher timeframe momentum. Volume confirmation filters false breakouts.
# Designed for low frequency (20-50 trades/year) to minimize fee drag. Works in bull/bear via trend filter.

name = "4h_1D_Camarilla_R1S1_Breakout_Trend_Volume"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 4h Camarilla levels (based on previous 4h bar)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12

    # Volume confirmation: current volume > 1.3x average of last 20 bars
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Daily trend filter
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price crosses above R1 with bullish daily trend and volume confirmation
            if close[i] > R1[i] and close[i-1] <= R1[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below S1 with bearish daily trend and volume confirmation
            elif close[i] < S1[i] and close[i-1] >= S1[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or daily trend turns bearish
            if close[i] < S1[i] and close[i-1] >= S1[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or daily trend turns bullish
            if close[i] > R1[i] and close[i-1] <= R1[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals