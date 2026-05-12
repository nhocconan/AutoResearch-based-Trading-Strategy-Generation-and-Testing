#!/usr/bin/env python3
"""
4h_DonchianBreakout_VolumeTrendFilter
Hypothesis: On 4h timeframe, buy when price breaks above Donchian upper band (20-period high) with volume >1.5x average and 1d EMA50 trending up; sell when price breaks below Donchian lower band (20-period low) with volume >1.5x average and 1d EMA50 trending down. Uses ADX(14) > 20 to filter ranging markets. Targets 20-40 trades per year to minimize fee drag and improve robustness in bull/bear markets.
"""

name = "4h_DonchianBreakout_VolumeTrendFilter"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res

    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res

    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # ADX(14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values using Wilder's smoothing
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result

        tr_smooth = smooth_wilder(tr, period)
        plus_dm_smooth = smooth_wilder(plus_dm, period)
        minus_dm_smooth = smooth_wilder(minus_dm, period)

        # DI+ and DI-
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        dx = np.where((plus_di + minus_di) != 0,
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
                      0)
        # ADX
        adx = smooth_wilder(dx, period)
        return adx

    adx_14 = calculate_adx(high, low, close, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)  # ADX uses same 1d alignment

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(adx_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + 1d uptrend + volume spike + ADX > 20
            if (close[i] > donchian_high[i] and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5 and
                adx_14_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + 1d downtrend + volume spike + ADX > 20
            elif (close[i] < donchian_low[i] and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5 and
                  adx_14_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR trend turns down OR ADX weakens
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i] or adx_14_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR trend turns up OR ADX weakens
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i] or adx_14_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals