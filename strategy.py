#!/usr/bin/env python3
# 6h_ADX_Volume_Trend_Filter
# Hypothesis: Use ADX(14) to identify strong trends (ADX > 25) on 6h, combined with +DI/-DI crossover for direction, confirmed by volume spikes (>1.5x 20-period average). Enter long when +DI crosses above -DI with ADX > 25 and volume spike; short when -DI crosses above +DI with ADX > 25 and volume spike. Exit when ADX falls below 20 (trend weakening) or reverse crossover. Targets 15-30 trades/year to avoid fee decay while capturing strong trends in both bull and bear markets.

name = "6h_ADX_Volume_Trend_Filter"
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

    # Calculate ADX and DI components
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]

    # Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0

    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values

    # DI values
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum

    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: +DI crosses above -DI with ADX > 25 and volume spike
            if (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1] and
                adx[i] > 25 and volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: -DI crosses above +DI with ADX > 25 and volume spike
            elif (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1] and
                  adx[i] > 25 and volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ADX < 20 (trend weakening) or -DI crosses above +DI
            if adx[i] < 20 or (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ADX < 20 (trend weakening) or +DI crosses above -DI
            if adx[i] < 20 or (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals