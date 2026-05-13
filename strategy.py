#!/usr/bin/env python3
# 4h_Choppiness_ADX_Trend_Reversal
# Hypothesis: In choppy markets (high Choppiness Index), price tends to reverse at Bollinger Bands extremes.
# In trending markets (ADX > 25), price follows ADX direction. Uses 1d EMA50 as higher timeframe filter to avoid counter-trend trades.
# Combines regime detection (Choppiness Index) with trend/follow logic for robustness in both bull and bear markets.
# Target: 20-30 trades/year per symbol.

name = "4h_Choppiness_ADX_Trend_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Choppiness Index (14-period)
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))

    atr14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    # ATR(14) using Wilder's smoothing (equivalent to RMA)
    atr14[13] = np.mean(tr[1:14])
    for i in range(14, n):
        atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14

    # Sum of true ranges over 14 periods
    sum_tr14 = np.zeros(n)
    for i in range(13, n):
        sum_tr14[i] = np.sum(tr[i-13:i+1])

    # Highest high and lowest low over 14 periods
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(n):
        if i < 13:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high[i-13:i+1])
            lowest_low[i] = np.min(low[i-13:i+1])

    # Choppiness Index: 100 * log10(sumTR14 / (n * ATR)) / log10(n)
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if atr14[i] > 0 and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(sum_tr14[i] / (atr14[i] * 14)) / np.log10(14)
        else:
            chop[i] = np.nan

    # ADX (14-period)
    # +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0

    # Smoothed +DM, -DM, TR (using Wilder's smoothing)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[1:period])  # Skip first for DM, but TR includes index 0
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result

    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    tr_smooth = wilders_smoothing(tr, 14)

    # +DI and -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    for i in range(14, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            plus_di[i] = np.nan
            minus_di[i] = np.nan
            dx[i] = np.nan

    # ADX: smoothed DX
    adx = wilders_smoothing(dx, 14)

    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup period for all indicators
        # Skip if any required value is NaN
        if (np.isnan(chop[i]) or np.isnan(adx[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG conditions
            long_condition = False
            # In choppy market (CHOP > 61.8): mean reversion at BB lower
            if chop[i] > 61.8:
                if close[i] <= bb_lower[i] and close[i] > bb_lower[i] * 0.999:  # Touch or slightly above lower band
                    long_condition = True
            # In trending market (ADX > 25): follow ADX direction if +DI > -DI
            elif adx[i] > 25:
                if plus_di[i] > minus_di[i] and close[i] > ema50_1d_aligned[i]:
                    long_condition = True
            # Transition zone: use price action with volume
            else:
                if close[i] > bb_mid[i] and volume[i] > vol_avg_20[i] * 1.3:
                    long_condition = True

            if long_condition:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0

        elif position == 1:
            # EXIT LONG conditions
            exit_long = False
            # In choppy market: exit at BB middle or upper
            if chop[i] > 61.8:
                if close[i] >= bb_mid[i]:
                    exit_long = True
            # In trending market: exit if ADX weakens or reversal
            elif adx[i] > 25:
                if plus_di[i] < minus_di[i] or close[i] < ema50_1d_aligned[i]:
                    exit_long = True
            # Transition zone: exit on mean reversion signals
            else:
                if close[i] < bb_mid[i]:
                    exit_long = True

            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25

        elif position == -1:
            # SHORT conditions (symmetrical to LONG)
            short_condition = False
            # In choppy market (CHOP > 61.8): mean reversion at BB upper
            if chop[i] > 61.8:
                if close[i] >= bb_upper[i] and close[i] < bb_upper[i] * 1.001:  # Touch or slightly below upper band
                    short_condition = True
            # In trending market (ADX > 25): follow ADX direction if -DI > +DI
            elif adx[i] > 25:
                if minus_di[i] > plus_di[i] and close[i] < ema50_1d_aligned[i]:
                    short_condition = True
            # Transition zone: use price action with volume
            else:
                if close[i] < bb_mid[i] and volume[i] > vol_avg_20[i] * 1.3:
                    short_condition = True

            if short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0

        elif position == -1:
            # EXIT SHORT conditions
            exit_short = False
            # In choppy market: exit at BB middle or lower
            if chop[i] > 61.8:
                if close[i] <= bb_mid[i]:
                    exit_short = True
            # In trending market: exit if ADX weakens or reversal
            elif adx[i] > 25:
                if minus_di[i] < plus_di[i] or close[i] > ema50_1d_aligned[i]:
                    exit_short = True
            # Transition zone: exit on mean reversion signals
            else:
                if close[i] > bb_mid[i]:
                    exit_short = True

            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals