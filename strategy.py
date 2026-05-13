#!/usr/bin/env python3
# 4h_KAMA_Trend_Filtered_Donchian_Breakout
# Hypothesis: Use KAMA(14) to determine trend direction on 4h, enter long on breakout above Donchian upper (20) when KAMA is rising, short on breakdown below Donchian lower when KAMA is falling. Filter entries with volume > 1.5x 20-period average and avoid choppy markets using Choppiness Index (14) > 61.8. Exit when price crosses KAMA. Designed to work in bull (breakouts with trend) and bear (breakdowns with trend) while avoiding false signals in ranging markets via chop filter.

name = "4h_KAMA_Trend_Filtered_Donchian_Breakout"
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

    # KAMA trend on 4h (ER=10, slow=2, fast=30)
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0  # first 10 values invalid
    vol = np.abs(np.diff(close, 1))
    vol = np.insert(vol, 0, 0)  # same length as close
    er = np.where(vol != 0, change / vol, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Donchian channels (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma

    # Choppiness Index (14) for regime filter
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high14 - lowest_low14) != 0, 
                    100 * np.log10(sum_tr14 / (highest_high14 - lowest_low14)) / np.log10(14), 
                    50)
    chop_filter = chop > 61.8  # only trade in choppy markets (mean reversion zone)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(kama[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_filter[i]) or np.isnan(chop_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: breakout above Donchian high + rising KAMA + volume filter + chop filter
            if close[i] > highest_high[i] and kama[i] > kama[i-1] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: breakdown below Donchian low + falling KAMA + volume filter + chop filter
            elif close[i] < lowest_low[i] and kama[i] < kama[i-1] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals