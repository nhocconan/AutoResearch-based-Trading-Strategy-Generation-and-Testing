#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: Use 4h Camarilla R1/S1 breakouts as directional signals in the direction of 1d EMA50 trend, with 1h volume confirmation.
# Entry: Break above R1 (long) or below S1 (short) on 1h, only if 4h trend aligns (price > 4h EMA50 for long, < 4h EMA50 for short) and 1h volume > 1.5x 20-period average.
# Exit: When price returns to Camarilla pivot level (mean reversion) or trend fails.
# Timeframe: 1h, using 4h for trend and Camarilla levels, 1d for volume context filter.
# Designed for low trade frequency (~20-50/year) with edge in both bull and bear markets via mean-reversion in trending environment.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    pivot = (high + low + close) / 3
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels
    camarilla_pivot_4h = np.zeros(len(df_4h))
    camarilla_r1_4h = np.zeros(len(df_4h))
    camarilla_s1_4h = np.zeros(len(df_4h))
    for i in range(len(df_4h)):
        pivot, r1, s1 = calculate_camarilla(df_4h['high'].iloc[i], df_4h['low'].iloc[i], df_4h['close'].iloc[i])
        camarilla_pivot_4h[i] = pivot
        camarilla_r1_4h[i] = r1
        camarilla_s1_4h[i] = s1
    
    camarilla_pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_4h)
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)

    # Get 1d data for volume context filter (average volume)
    df_1d = get_htf_data(prices, '1d')
    avg_volume_1d = pd.Series(df_1d['volume']).mean().values  # scalar average daily volume
    # Use scalar for comparison - we want today's volume to be above average daily volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_4h_aligned[i]) or 
            np.isnan(camarilla_s1_4h_aligned[i]) or np.isnan(camarilla_pivot_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volume filter: current volume > 1.5x average daily volume (scalar)
        volume_filter = volume[i] > (avg_volume_1d * 1.5)

        if position == 0:
            # LONG: Break above R1 + 4h uptrend (price > EMA50) + volume filter
            if (close[i] > camarilla_r1_4h_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and
                volume_filter):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 + 4h downtrend (price < EMA50) + volume filter
            elif (close[i] < camarilla_s1_4h_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  volume_filter):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Return to pivot level (mean reversion) or trend fails
            if (close[i] <= camarilla_pivot_4h_aligned[i] or close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Return to pivot level (mean reversion) or trend fails
            if (close[i] >= camarilla_pivot_4h_aligned[i] or close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals