#!/usr/bin/env python3
# 6h_Williams_Alligator_ElderRay_1dTrend
# Hypothesis: Combine Williams Alligator (trend detection) with Elder Ray (bull/bear power) filtered by 1d EMA trend direction.
# Alligator jaws/teeth/lips define trend state; Elder Ray power confirms bull/bear strength.
# Enter long when Elder Bull Power > 0 and price above Alligator teeth in uptrend (1d EMA50 up).
# Enter short when Elder Bear Power < 0 and price below Alligator teeth in downtrend (1d EMA50 down).
# Designed for 6h timeframe with low trade frequency (target: 50-150 total trades over 4 years).

name = "6h_Williams_Alligator_ElderRay_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) used in Williams Alligator."""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    smma = np.full_like(arr, np.nan, dtype=float)
    smma[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        smma[i] = (smma[i-1] * (period-1) + arr[i]) / period
    return smma

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
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_50_1d_up = ema_50_1d > ema_50_1d_prev
    ema_50_1d_down = ema_50_1d < ema_50_1d_prev
    ema_50_1d_up = align_htf_to_ltf(prices, df_1d, ema_50_1d_up)
    ema_50_1d_down = align_htf_to_ltf(prices, df_1d, ema_50_1d_down)

    # Williams Alligator (13,8,5 SMMA with 8,5,3 offsets)
    jaws = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Apply offsets: jaws+8, teeth+5, lips+3
    jaws = np.roll(jaws, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Fill rolled NaNs
    jaws[:8] = jaws[8] if not np.isnan(jaws[8]) else np.nan
    teeth[:5] = teeth[5] if not np.isnan(teeth[5]) else np.nan
    lips[:3] = lips[3] if not np.isnan(lips[3]) else np.nan

    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_up[i]) or np.isnan(ema_50_1d_down[i]) or
            np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 AND price above teeth (bullish alignment) AND 1d EMA50 trending up
            if (bull_power[i] > 0 and 
                close[i] > teeth[i] and
                ema_50_1d_up[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND price below teeth (bearish alignment) AND 1d EMA50 trending down
            elif (bear_power[i] < 0 and 
                  close[i] < teeth[i] and
                  ema_50_1d_down[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power >= 0 (loss of bullish momentum) OR price below lips (trend weakening)
            if (bear_power[i] >= 0 or close[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power <= 0 (loss of bearish momentum) OR price above lips (trend weakening)
            if (bull_power[i] <= 0 or close[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals