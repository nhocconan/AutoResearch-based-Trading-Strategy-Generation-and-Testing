#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Confirmation
Hypothesis: On 1d timeframe, KAMA (Kaufman Adaptive Moving Average) adapts to market noise,
providing smoother trend signals in both trending and ranging markets. Combined with
volume confirmation (>1.5x average) and ADX filter (>20) to avoid false signals.
Targets 10-20 trades per year to minimize fee drag and improve generalization.
Works in bull markets (trend following) and bear markets (avoids false breakouts).
"""

name = "1d_KAMA_Trend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Handle array case properly
    if len(close) > 1:
        volatility = np.abs(np.diff(close))
        volatility_sum = np.zeros_like(close)
        for i in range(len(close)):
            start = max(0, i - length + 1)
            volatility_sum[i] = np.sum(volatility[start:i+1])
        er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    else:
        er = np.zeros_like(close)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate KAMA on 1d (using close prices)
    kama = calculate_kama(close, length=10, fast=2, slow=30)

    # Calculate 1w EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # ADX(14) for trend strength filter
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)
    
    # Calculate DI+ and DI-
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = smooth_wilder(dx, 14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + 1w uptrend + volume confirmation + ADX > 20
            if (close[i] > kama[i] and 
                close[i] > ema20_1w_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and
                adx[i] > 20):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + 1w downtrend + volume confirmation + ADX > 20
            elif (close[i] < kama[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and
                  adx[i] > 20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA OR 1w trend turns down OR ADX weakens
            if close[i] < kama[i] or close[i] < ema20_1w_aligned[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR 1w trend turns up OR ADX weakens
            if close[i] > kama[i] or close[i] > ema20_1w_aligned[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals