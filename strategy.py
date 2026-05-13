#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_Trend_Reversal
# Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) identifies
# trend exhaustion when power weakens while price makes new highs/lows (divergence).
# Long when Bull Power turns up from negative while price > EMA13 (bullish reversal).
# Short when Bear Power turns up from negative while price < EMA13 (bearish reversal).
# Uses 1-day ADX > 20 to confirm trending environment and avoid whipsaws in ranging markets.
# Target: 15-30 trades/year (~60-120 total over 4 years) with controlled risk.

name = "6h_ElderRay_BullBearPower_Trend_Reversal"
timeframe = "6h"
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

    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low

    # Previous values for crossover detection
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = np.nan
    bear_power_prev[0] = np.nan

    # Get 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = np.nan
        tr2[0] = np.nan
        tr3[0] = np.nan
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = np.nan
        down_move[0] = np.nan
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        def WilderSmoothing(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) < period:
                return result
            # First value: simple average
            result[period-1] = np.nansum(arr[:period]) / period
            # Subsequent values: Wilder smoothing
            for i in range(period, len(arr)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i] / period
            return result

        atr = WilderSmoothing(tr, period)
        plus_di = 100 * WilderSmoothing(plus_dm, period) / atr
        minus_di = 100 * WilderSmoothing(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = WilderSmoothing(dx, period)
        return adx

    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # Volume confirmation: above 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_prev[i]) or np.isnan(bear_power_prev[i]) or
            np.isnan(ema13[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: only trade when 1d ADX > 20 (trending market)
        if adx_1d_aligned[i] <= 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power turning up from negative AND price > EMA13
            if (bull_power[i] > bull_power_prev[i] and 
                bull_power[i] < 0 and  # still negative but improving
                close[i] > ema13[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power turning up from negative AND price < EMA13
            elif (bear_power[i] > bear_power_prev[i] and 
                  bear_power[i] < 0 and  # still negative but improving
                  close[i] < ema13[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns down OR price crosses below EMA13
            if (bull_power[i] < bull_power_prev[i] or close[i] < ema13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns down OR price crosses above EMA13
            if (bear_power[i] < bear_power_prev[i] or close[i] > ema13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals