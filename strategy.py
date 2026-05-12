# [160895] 6h_1w_1d_Camarilla_R3S3_Breakout_Volume_Trend
# Hypothesis: 6-hour breakouts above weekly R3 or below weekly S3 with daily volume confirmation and weekly trend filter.
# Uses weekly trend to filter out noise and avoid whipsaws in range markets. Volume confirms institutional interest.
# Designed for 12-37 trades per year (50-150 over 4 years) to minimize fee drag.
# Works in bull markets (follow weekly trend up) and bear markets (follow weekly trend down) by aligning with higher timeframe momentum.

name = "6h_1w_1d_Camarilla_R3S3_Breakout_Volume_Trend"
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

    # Get weekly data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate weekly Camarilla levels (R3 and S3) based on previous week
    prev_high = np.roll(df_1w['high'].values, 1)
    prev_low = np.roll(df_1w['low'].values, 1)
    prev_close = np.roll(df_1w['close'].values, 1)
    prev_high[0] = df_1w['high'].values[0]
    prev_low[0] = df_1w['low'].values[0]
    prev_close[0] = df_1w['close'].values[0]
    
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4

    # Align weekly levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)

    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate daily volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volume confirmation: current 6h volume > 1.5x daily average volume (scaled)
        # Approximate: compare 6h volume to 1/4 of daily volume (since 4x 6h = 1d)
        vol_scaled = vol_ma_1d_aligned[i] * 0.25  # Scale daily MA to 6h equivalent
        volume_ok = volume[i] > (1.5 * vol_scaled)

        # Weekly trend filter
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Price closes above R3 with bullish weekly trend and volume confirmation
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] and bullish_trend and volume_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with bearish weekly trend and volume confirmation
            elif close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] and bearish_trend and volume_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 or weekly trend turns bearish
            if close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 or weekly trend turns bullish
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals