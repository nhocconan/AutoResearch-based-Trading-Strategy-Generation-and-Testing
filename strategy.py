#!/usr/bin/env python3
# 4h_RVI_Slope_Filter
# Hypothesis: Use Relative Vigor Index (RVI) slope to detect trend momentum on 4h, confirmed by 1d trend (EMA50) and volume spikes (>2x 20-period average). Enter long when RVI slope > 0 and price > 1d EMA50 with volume spike; short when RVI slope < 0 and price < 1d EMA50 with volume spike. Exit on RVI slope cross. Targets 20-50 trades/year to minimize fee drag and work in both bull/bear markets via trend filter.

name = "4h_RVI_Slope_Filter"
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
    open_ = prices['open'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Relative Vigor Index (RVI) on 4h
    numerator = close - open_
    denominator = high - low
    # Avoid division by zero
    denom_safe = np.where(denominator == 0, 1e-10, denominator)
    rvi_raw = numerator / denom_safe

    # Smooth numerator and denominator with 4-period SMA
    num_smooth = pd.Series(rvi_raw).rolling(window=4, min_periods=4).mean().values
    den_smooth = pd.Series(denominator).rolling(window=4, min_periods=4).mean().values
    den_smooth_safe = np.where(den_smooth == 0, 1e-10, den_smooth)
    rvi = num_smooth / den_smooth_safe

    # Calculate RVI slope (4-period difference)
    rvi_slope = rvi - np.roll(rvi, 4)
    rvi_slope[:4] = 0  # First 4 values invalid

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(rvi_slope[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RVI slope > 0 (bullish momentum) + price > 1d EMA50 + volume spike
            if (rvi_slope[i] > 0 and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: RVI slope < 0 (bearish momentum) + price < 1d EMA50 + volume spike
            elif (rvi_slope[i] < 0 and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RVI slope turns negative
            if rvi_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RVI slope turns positive
            if rvi_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals