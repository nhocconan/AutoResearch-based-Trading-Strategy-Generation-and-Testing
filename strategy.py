#!/usr/bin/env python3
# 4h_Vortex_Volume_Trend_Filter
# Hypothesis: Vortex indicator (VI+ and VI-) identifies trend direction while volume confirms momentum.
# In bull markets: VI+ > VI- with volume surge = long. In bear markets: VI- > VI+ with volume surge = short.
# Uses 1d trend filter (EMA50) to avoid counter-trend trades. Target: 20-50 trades/year to minimize fee drag.

name = "4h_Vortex_Volume_Trend_Filter"
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

    # Vortex Indicator (VI) - 14 period
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = 0
    vm = np.abs(high - np.roll(low, 1))  # +VM
    vm[0] = 0
    vp = np.abs(low - np.roll(high, 1))   # -VM
    vp[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm14 = pd.Series(vm).rolling(window=14, min_periods=14).sum().values
    vp14 = pd.Series(vp).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm14 / tr14
    vi_minus = vp14 / tr14

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ > VI- (bullish) + above 1d EMA50 + volume surge
            if (vi_plus[i] > vi_minus[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish) + below 1d EMA50 + volume surge
            elif (vi_minus[i] > vi_plus[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- > VI+ (trend change) or volume drop
            if vi_minus[i] > vi_plus[i] or volume[i] < vol_avg_20[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ > VI- (trend change) or volume drop
            if vi_plus[i] > vi_minus[i] or volume[i] < vol_avg_20[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals