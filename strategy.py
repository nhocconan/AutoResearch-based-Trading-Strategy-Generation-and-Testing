#!/usr/bin/env python3
# 4h_ADX_Strength_Breakout
# Hypothesis: Use ADX to filter trending markets (ADX > 25) and breakouts from 20-period Donchian channels for entry.
# In trending markets, breakouts are more likely to continue; in ranging markets (ADX < 20), avoid false breakouts.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Designed for low trade frequency to minimize fee drag and work in both bull/bear markets via ADX trend filter.

name = "4h_ADX_Strength_Breakout"
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

    # Calculate ADX (14-period) for trend strength filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[:1] = 0  # First value has no previous close

    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result

    period = 14
    tr_sum = wilder_smooth(tr, period)
    plus_dm_sum = wilder_smooth(plus_dm, period)
    minus_dm_sum = wilder_smooth(minus_dm, period)

    # Avoid division by zero
    tr_sum_safe = np.where(tr_sum == 0, 1e-10, tr_sum)
    plus_di = 100 * plus_dm_sum / tr_sum_safe
    minus_di = 100 * minus_dm_sum / tr_sum_safe
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx, period)

    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(adx[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: ADX > 25 (trending) + price breaks above upper Donchian + volume surge
            if (adx[i] > 25 and 
                close[i] > highest_high[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: ADX > 25 (trending) + price breaks below lower Donchian + volume surge
            elif (adx[i] > 25 and 
                  close[i] < lowest_low[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel (below midpoint) OR ADX weakens (< 20)
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel (above midpoint) OR ADX weakens (< 20)
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals