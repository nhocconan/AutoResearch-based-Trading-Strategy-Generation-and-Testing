#!/usr/bin/env python3
"""
4h_Triple_Confirmation_Breakout
Hypothesis: Combines Donchian breakout (20-period) with ADX trend strength and volume confirmation to capture strong trending moves. Uses ADX > 25 to filter choppy markets, volume > 1.5x 20-period average for confirmation, and Donchian breakouts for entry. Designed for 4h timeframe to balance trade frequency and signal quality, targeting 20-40 trades per year. Works in both bull and bear markets by requiring strong trend confirmation (ADX) before entering breakout trades.
"""

name = "4h_Triple_Confirmation_Breakout"
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

    # Donchian Channel (20-period) - using pandas rolling with min_periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # ADX calculation (14-period) - measures trend strength
    # True Range
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length

    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])

    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smoothing(arr, period):
        """Wilder's smoothing (same as EMA with alpha=1/period)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])  # skip first NaN
        # Subsequent values: smoothed = prev * (1-1/period) + current * (1/period)
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]):
                if np.isnan(result[i-1]):
                    result[i] = arr[i]
                else:
                    result[i] = result[i-1] * (1 - 1/period) + arr[i] * (1/period)
        return result

    tr14 = wilder_smoothing(tr, 14)
    dm_plus_14 = wilder_smoothing(dm_plus, 14)
    dm_minus_14 = wilder_smoothing(dm_minus, 14)

    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)

    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smoothing(dx, 14)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # start from 20 to ensure Donchian is valid
        # Skip if any required values are NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Donchian high + ADX > 25 (strong trend) + volume confirmation
            if (close[i] > donchian_high[i] and 
                adx[i] > 25 and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low + ADX > 25 (strong trend) + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  adx[i] > 25 and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low or ADX falls below 20 (trend weakening)
            if close[i] < donchian_low[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high or ADX falls below 20 (trend weakening)
            if close[i] > donchian_high[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals