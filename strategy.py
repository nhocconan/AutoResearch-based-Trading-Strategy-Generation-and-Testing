#!/usr/bin/env python3
# 4h_HTF_Pivot_Breakout_Volume_Trend
# Hypothesis: On 4h timeframe, buy when price breaks above daily pivot point with volume >1.5x average and 1d EMA50 trending up; sell when price breaks below daily pivot point with volume >1.5x average and 1d EMA50 trending down. Uses ADX(14) > 20 trend filter to reduce false breakouts in ranging markets. Pivot point provides objective support/resistance, volume confirms breakout strength, and trend filter ensures alignment with higher timeframe momentum. Targets 20-40 trades per year to minimize fee drag and improve generalization across bull/bear markets.

name = "4h_HTF_Pivot_Breakout_Volume_Trend"
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

    # Get 1d data for pivot point and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot point: P = (H + L + C) / 3
    pivot_point = (high_1d + low_1d + close_1d) / 3.0

    # Use previous day's pivot (shift by 1) to avoid look-ahead
    pivot_prev = np.roll(pivot_point, 1)
    pivot_prev[0] = np.nan

    # Align pivot point to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_prev)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

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
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = smooth_wilder(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Volume confirmation: volume > 1.5x 20-period average (approx 10 hours)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above pivot + 1d uptrend + volume spike + ADX > 20
            if (close[i] > pivot_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and
                adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below pivot + 1d downtrend + volume spike + ADX > 20
            elif (close[i] < pivot_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and
                  adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below pivot OR trend turns down OR ADX weakens
            if close[i] < pivot_aligned[i] or close[i] < ema50_1d_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above pivot OR trend turns up OR ADX weakens
            if close[i] > pivot_aligned[i] or close[i] > ema50_1d_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals