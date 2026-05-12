#!/usr/bin/env python3
# 4h_VolumeBreakout_12hTrend
# Hypothesis: Break above 12-period high on 4h with volume >1.5x 24-period avg and 12h EMA50 trending up; break below 12-period low with volume >1.5x 24-period avg and 12h EMA50 trending down. Uses ADX(14)>25 on 12h to filter ranging markets. Targets 20-40 trades/year to minimize fee drift. Works in bull/bear via trend filter.

name = "4h_VolumeBreakout_12hTrend"
timeframe = "4h"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # 12h ADX(14) for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values

    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h_arr, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h_arr, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # +DM and -DM
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result

    tr_smooth = smooth_wilder(tr, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)

    # DI and DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = np.where((plus_di + minus_di) != 0,
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
                  0)
    adx_12h = smooth_wilder(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)

    # 4h breakout levels: 12-period high/low
    high_roll = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_roll = pd.Series(low).rolling(window=12, min_periods=12).min().values

    # Volume confirmation: volume > 1.5x 24-period average
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_24[i]) or
            np.isnan(adx_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above 12-period high + volume spike + 12h uptrend + ADX > 25
            if (close[i] > high_roll[i] and
                volume[i] > vol_avg_24[i] * 1.5 and
                close[i] > ema50_12h_aligned[i] and
                adx_12h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 12-period low + volume spike + 12h downtrend + ADX > 25
            elif (close[i] < low_roll[i] and
                  volume[i] > vol_avg_24[i] * 1.5 and
                  close[i] < ema50_12h_aligned[i] and
                  adx_12h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below 12-period low OR trend turns down OR ADX weakens
            if close[i] < low_roll[i] or close[i] < ema50_12h_aligned[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above 12-period high OR trend turns up OR ADX weakens
            if close[i] > high_roll[i] or close[i] > ema50_12h_aligned[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals