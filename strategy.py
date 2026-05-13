#!/usr/bin/env python3
# 1d_Choppiness_Index_Regime + Donchian(20) Breakout + Volume Confirmation
# Hypothesis: Choppiness Index (CHOP) filters market regime. In trending markets (CHOP < 38.2), we trade Donchian breakouts; in ranging markets (CHOP > 61.8), we avoid trades. This avoids whipsaws in sideways markets and captures trends in both bull and bear regimes. Volume confirmation ensures breakout validity. Designed for low trade frequency (target: 10-25/year) to minimize fee drag.

name = "1d_Choppiness_Index_Regime + Donchian(20) Breakout + Volume Confirmation"
timeframe = "1d"
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

    # Choppiness Index (CHOP) - 14 period
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))

    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]  # first TR
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])

    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values

    chop = np.full(n, np.nan)
    for i in range(14, n):
        if atr14[i] > 0 and highest_high14[i] > lowest_low14[i]:
            chop[i] = 100 * np.log14(np.sum(tr[i-13:i+1]) / (atr14[i] * 14)) / np.log14(14)
        else:
            chop[i] = 50.0  # neutral if invalid

    # Donchian(20) channels
    highest_high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Weekly trend filter: EMA50 on 1w
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # warmup for CHOP(14) + Donchian(20)
        # Skip if any required value is NaN
        if (np.isnan(chop[i]) or np.isnan(highest_high20[i]) or np.isnan(lowest_low20[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Chop < 38.2 (trending) + close > Donchian High + volume spike + price > weekly EMA50 (uptrend)
            if (chop[i] < 38.2 and
                close[i] > highest_high20[i] and
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Chop < 38.2 (trending) + close < Donchian Low + volume spike + price < weekly EMA50 (downtrend)
            elif (chop[i] < 38.2 and
                  close[i] < lowest_low20[i] and
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chop > 61.8 (ranging) OR close < Donchian Low
            if (chop[i] > 61.8 or close[i] < lowest_low20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chop > 61.8 (ranging) OR close > Donchian High
            if (chop[i] > 61.8 or close[i] > highest_high20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals