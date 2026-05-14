#!/usr/bin/env python3
# 4h_Vortex_Volume_Trend_Filter
# Hypothesis: Use Vortex Indicator (VI+) and (VI-) to detect trend direction on 4h, confirmed by 1d trend (EMA50) and volume spikes (>2x 20-period average). Enter long when VI+ > VI- and price > 1d EMA50 with volume spike; short when VI- > VI+ and price < 1d EMA50 with volume spike. Exit on Vortex crossover reverse. Targets 20-50 trades/year to minimize fee drag and work in both bull/bear markets via trend filter.

name = "4h_Vortex_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Vortex Indicator (VI) on 4h
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first bar true range

    # Vortex Movement
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.abs(high[0] - low[0])
    vm_minus[0] = np.abs(low[0] - high[0])

    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values / \
              pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values / \
               pd.Series(tr).rolling(window=14, min_periods=14).sum().values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
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
            # LONG: VI+ > VI- (bullish vortex) + price > 1d EMA50 + volume spike
            if (vi_plus[i] > vi_minus[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish vortex) + price < 1d EMA50 + volume spike
            elif (vi_minus[i] > vi_plus[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Vortex turns bearish (VI- > VI+)
            if vi_minus[i] > vi_plus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Vortex turns bullish (VI+ > VI-)
            if vi_plus[i] > vi_minus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals