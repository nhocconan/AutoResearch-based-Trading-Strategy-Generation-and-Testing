#!/usr/bin/env python3
"""
4h_RSI40_60_With_Volume_and_ADX_Filter
Hypothesis: Buy when RSI crosses above 40 with volume confirmation and ADX > 25 (trending market); sell when RSI crosses below 60 with volume confirmation and ADX > 25. Uses 1d ADX for regime filter to avoid whipsaws in sideways markets. Works in both bull and bear by only taking trades in the direction of the 1d trend (ADX > 25 indicates trending, direction from price vs EMA50).
"""

name = "4h_RSI40_60_With_Volume_and_ADX_Filter"
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

    # Get 1d data ONCE before loop for ADX and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan  # First value has no previous close

        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = np.nan
        down_move[0] = np.nan

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
        def WilderSmoothing(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(arr[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result

        tr_smooth = WilderSmoothing(tr, period)
        plus_dm_smooth = WilderSmoothing(plus_dm, period)
        minus_dm_smooth = WilderSmoothing(minus_dm, period)

        # Avoid division by zero
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)

        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = WilderSmoothing(dx, period)
        return adx

    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)

    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # RSI(14) on 4h
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=np.nan)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # Wilder's smoothing
        def WilderSmoothing(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            result[period-1] = np.nanmean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result

        avg_gain = WilderSmoothing(gain, period)
        avg_loss = WilderSmoothing(loss, period)

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_14 = calculate_rsi(close, 14)

    # Volume confirmation: >1.3x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 and ADX warmup
        if (np.isnan(rsi_14[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI crosses above 40 + volume spike + ADX > 25 + price above EMA50 (uptrend)
            if (rsi_14[i] > 40 and rsi_14[i-1] <= 40 and  # Cross above 40
                volume_spike[i] and
                adx_14_1d_aligned[i] > 25 and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below 60 + volume spike + ADX > 25 + price below EMA50 (downtrend)
            elif (rsi_14[i] < 60 and rsi_14[i-1] >= 60 and  # Cross below 60
                  volume_spike[i] and
                  adx_14_1d_aligned[i] > 25 and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below 50 (momentum fade)
            if rsi_14[i] < 50 and rsi_14[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses above 50 (momentum fade)
            if rsi_14[i] > 50 and rsi_14[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals