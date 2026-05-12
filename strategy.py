#!/usr/bin/env python3

# 6h_1D_ADX_DMI_Trend_Filter
# Hypothesis: Use ADX(14) and +DI/-DI crossover on daily timeframe to filter 6h momentum entries.
# In trending markets (ADX > 25), enter long when +DI crosses above -DI, short when -DI crosses above +DI.
# Exit when ADX falls below 20 or reverse crossover occurs. This avoids whipsaws in ranging markets.
# Works in both bull and bear by following the dominant trend on higher timeframe.

name = "6h_1D_ADX_DMI_Trend_Filter"
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

    # Get daily data for ADX/DMI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr])

    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])

    # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    period = 14
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)

    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth

    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = wilder_smooth(dx, period)

    # Align to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or 
            np.isnan(di_minus_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        adx_val = adx_aligned[i]
        di_plus_val = di_plus_aligned[i]
        di_minus_val = di_minus_aligned[i]

        # Previous values for crossover detection
        if i > 0:
            di_plus_prev = di_plus_aligned[i-1]
            di_minus_prev = di_minus_aligned[i-1]
        else:
            di_plus_prev = di_plus_val
            di_minus_prev = di_minus_val

        if position == 0:
            # ENTER LONG: ADX > 25 and +DI crosses above -DI
            if (adx_val > 25 and 
                di_plus_val > di_minus_val and 
                di_plus_prev <= di_minus_prev):
                signals[i] = 0.25
                position = 1
            # ENTER SHORT: ADX > 25 and -DI crosses above +DI
            elif (adx_val > 25 and 
                  di_minus_val > di_plus_val and 
                  di_minus_prev <= di_plus_prev):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ADX < 20 or -DI crosses above +DI
            if (adx_val < 20 or 
                (di_minus_val > di_plus_val and di_minus_prev <= di_plus_prev)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ADX < 20 or +DI crosses above -DI
            if (adx_val < 20 or 
                (di_plus_val > di_minus_val and di_plus_prev <= di_minus_prev)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals