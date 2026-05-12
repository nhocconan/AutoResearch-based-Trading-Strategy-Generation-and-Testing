#!/usr/bin/env python3
"""
12h_KAMA_Trend_Plus_Volume_Spike
Hypothesis: KAMA adapts to market noise—trend in low volatility, range-bound in high.
On 12h, go long when KAUMA turns up (bullish trend) with volume spike (>2x avg),
short when KAMA turns down (bearish trend) with volume spike.
Use 1d ADX < 20 to filter ranging markets (avoid false signals).
Targets 12-37 trades/year (50-150 total over 4 years) with low turnover.
Works in bull via trend continuation, bear via mean-reversion at extremes with volume filter.
"""

name = "12h_KAMA_Trend_Plus_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d ADX(14) for trend strength filter
    # ADX components: +DI, -DI, DX
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first value

        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values
        def smooth(values, period):
            smoothed = np.zeros_like(values)
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
            return smoothed

        tr_sum = smooth(tr, period)
        plus_dm_sum = smooth(plus_dm, period)
        minus_dm_sum = smooth(minus_dm, period)

        # Avoid division by zero
        plus_di = np.where(tr_sum != 0, (plus_dm_sum / tr_sum) * 100, 0)
        minus_di = np.where(tr_sum != 0, (minus_dm_sum / tr_sum) * 100, 0)

        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0,
                      np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = smooth(dx, period)
        return adx

    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # Calculate KAMA on 12h close
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_period))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
        # Manual rolling sum for volatility
        vol_sum = np.zeros_like(close)
        for i in range(len(close)):
            start = max(0, i - er_period + 1)
            vol_sum[i] = np.sum(np.abs(np.diff(close[start:i+1], prepend=close[start])))
        er = np.where(vol_sum != 0, change / vol_sum, 0)

        # Smoothing Constant
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2

        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama = calculate_kama(close, 10, 2, 30)
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama[0]
    kama_dir = np.where(kama > kama_prev, 1, np.where(kama < kama_prev, -1, 0))

    # Volume confirmation: 2.0x 20-period average
    vol_avg_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        start = max(0, i - 20 + 1)
        vol_avg_20[i] = np.mean(volume[start:i+1]) if (i - start + 1) >= 20 else np.nan

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # wait for KAMA and ADX warmup
        # Get aligned values for current 12h bar
        adx = adx_1d_aligned[i]
        kama_direction = kama_dir[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(adx) or np.isnan(kama_direction) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Range filter: only trade when ADX < 20 (ranging market)
        if adx >= 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA turning up + volume spike
            if (kama_direction == 1 and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down + volume spike
            elif (kama_direction == -1 and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turning down
            if kama_direction == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turning up
            if kama_direction == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals