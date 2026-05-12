#!/usr/bin/env python3
"""
12h_TRIX_Signal_With_Volume_Confirmation
Hypothesis: On 12h timeframe, use TRIX (triple smoothed EMA) as primary momentum signal combined with volume confirmation and ADX trend filter. Long when TRIX crosses above zero with volume > 2x average and ADX > 20; short when TRIX crosses below zero with volume > 2x average and ADX > 20. Exit on opposite TRIX cross. Targets 20-40 trades per year to reduce fee drag and improve generalization in bull/bear markets.
"""

name = "12h_TRIX_Signal_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for TRIX and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate TRIX (15,9,9) on 1d close
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9) - then percent change
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_raw = pd.Series(ema3).pct_change() * 100  # TRIX as percentage
    trix = trix_raw.fillna(0).values

    # 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing
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
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    # Avoid division by zero
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = smooth_wilder(dx, 14)

    # Align TRIX and ADX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Volume confirmation: volume > 2x 24-period average (approx 12 hours)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume spike + ADX > 20
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume[i] > vol_avg_24[i] * 2.0 and
                adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike + ADX > 20
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume[i] > vol_avg_24[i] * 2.0 and
                  adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals