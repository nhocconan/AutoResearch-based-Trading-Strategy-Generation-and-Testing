#!/usr/bin/env python3
# 6h_Pivot_Reversal_Volume
# Hypothesis: Fade at daily pivot support/resistance with volume confirmation on 6h timeframe.
# Uses daily pivot points (PP, R1, S1, R2, S2) calculated from prior day's OHLC.
# Long when price bounces off S1/S2 with bullish volume (close > open and volume > 1.5x average).
# Short when price rejects at R1/R2 with bearish volume (close < open and volume > 1.5x average).
# Designed for low trade frequency (12-37/year) to avoid fee flood. Works in ranging markets.
# Exit when price crosses the daily pivot point (PP) in the opposite direction.

name = "6h_Pivot_Reversal_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values

    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points: based on prior day's OHLC
    # Pivot Point (PP) = (high + low + close) / 3
    # Support 1 (S1) = (2 * PP) - high
    # Resistance 1 (R1) = (2 * PP) - low
    # Support 2 (S2) = PP - (high - low)
    # Resistance 2 (R2) = PP + (high - low)
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = (2 * pp) - high_1d
    s1 = (2 * pp) - low_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)

    # Align pivot points to 6h timeframe (use prior day's levels for current day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)

    # Volume filter: 1.5x 20-period SMA on 6h
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: bounce off S1 or S2 with bullish candle and volume
            bullish = close[i] > open_[i]
            vol_ok = volume[i] > volume_filter[i]
            near_s1 = low[i] <= s1_aligned[i] * 1.001  # allow small tolerance
            near_s2 = low[i] <= s2_aligned[i] * 1.001
            if (near_s1 or near_s2) and bullish and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: rejection at R1 or R2 with bearish candle and volume
            elif (not bullish) and vol_ok:
                near_r1 = high[i] >= r1_aligned[i] * 0.999
                near_r2 = high[i] >= r2_aligned[i] * 0.999
                if near_r1 or near_r2:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below pivot point
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals