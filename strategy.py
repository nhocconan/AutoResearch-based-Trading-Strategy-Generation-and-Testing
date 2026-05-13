#!/usr/bin/env python3
# 1d_Vortex_Volume_Trend_Filter
# Hypothesis: Use Vortex indicator on weekly trend filter with daily price action.
# Enter long when VI+ crosses above VI- on weekly timeframe AND price breaks above daily EMA20 with volume confirmation.
# Enter short when VI- crosses above VI+ on weekly timeframe AND price breaks below daily EMA20 with volume confirmation.
# Exit when price crosses back below/above EMA20.
# Vortex helps identify trend direction and reduces whipsaw in sideways markets.
# Combined with daily EMA trend and volume filter to capture strong moves while avoiding chop.
# Target: 10-25 trades/year on 1d to minimize fee drag while capturing sustained trends.

name = "1d_Vortex_Volume_Trend_Filter"
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

    # Get weekly data for Vortex trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate Vortex indicator (VI+ and VI-) on weekly data
    period = 14
    tr = np.maximum(
        np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1])),
        np.abs(low_1w[1:] - close_1w[:-1])
    )
    tr = np.concatenate([[np.nan], tr])  # Align length

    vm_plus = np.abs(high_1w[1:] - low_1w[:-1])
    vm_minus = np.abs(low_1w[1:] - high_1w[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])

    # Sum over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values

    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum

    # Align Vortex indicators to daily timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1w, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1w, vi_minus)

    # Daily EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ crosses above VI- AND price > EMA20 AND volume spike
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                vi_plus_aligned[i-1] <= vi_minus_aligned[i-1] and
                close[i] > ema20[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+ AND price < EMA20 AND volume spike
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                  vi_minus_aligned[i-1] <= vi_plus_aligned[i-1] and
                  close[i] < ema20[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA20
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA20
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals