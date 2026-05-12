#!/usr/bin/env python3
"""
12h_KAMA_Direction_With_Volume_Confirmation
Hypothesis: On 12h timeframe, use Kaufman's Adaptive Moving Average (KAMA) to capture trend direction with reduced whipsaw. Enter long when price > KAMA and volume > 1.5x average, short when price < KAMA and volume > 1.5x average. Exit when price crosses back across KAMA. Uses 1d ADX > 20 to avoid ranging markets. Targets 20-40 trades per year to minimize fee drag while maintaining trend capture in both bull and bear markets.
"""

name = "12h_KAMA_Direction_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_ema=2, slow_ema=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-er_length+1):i+1])))
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate KAMA on close prices
    kama = calculate_kama(close, er_length=10, fast_ema=2, slow_ema=30)

    # 1d ADX(14) for trend strength filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)
    
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = smooth_wilder(dx, 14)
    adx_1d = adx  # already calculated on 1d data
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_avg_20[:10] = np.nan  # insufficient data at start
    vol_avg_20[-10:] = np.nan

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + volume spike + ADX > 20
            if (close[i] > kama[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and
                adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + volume spike + ADX > 20
            elif (close[i] < kama[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and
                  adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals