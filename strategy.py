#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike
# Hypothesis: 6-hour breakouts above 12-hour R3 or below 12-hour S3 with volume confirmation and 12-hour trend filter.
# Uses 12-hour timeframe for trend and pivot levels to balance signal quality and trade frequency.
# Designed for 50-150 total trades over 4 years (12-37/year). Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by following the 12-hour trend.

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
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

    # Get 12-hour data for trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # Calculate 12-hour EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # Calculate 12-hour Camarilla levels (R3 and S3) based on previous 12-hour bar
    prev_high = np.roll(df_12h['high'].values, 1)
    prev_low = np.roll(df_12h['low'].values, 1)
    prev_close = np.roll(df_12h['close'].values, 1)
    prev_high[0] = df_12h['high'].values[0]
    prev_low[0] = df_12h['low'].values[0]
    prev_close[0] = df_12h['close'].values[0]
    
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4

    # Align 12-hour levels to 6-hour timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # 12-hour trend filter
        bullish_trend = close[i] > ema_12h_aligned[i]
        bearish_trend = close[i] < ema_12h_aligned[i]

        if position == 0:
            # LONG: Price closes above R3 with bullish 12h trend and volume confirmation
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with bearish 12h trend and volume confirmation
            elif close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 or 12h trend turns bearish
            if close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 or 12h trend turns bullish
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals