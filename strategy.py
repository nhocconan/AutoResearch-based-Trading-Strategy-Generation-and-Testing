#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_1dVolume
Hypothesis: Trade in the direction of weekly pivot point trend (price above/below weekly pivot) 
on 6-hour timeframe with daily volume confirmation. Weekly pivot provides institutional 
reference point; price above indicates bullish bias, below bearish bias. Volume spike 
confirms participation. Works in bull/bear by following weekly structure. Targets 12-30 trades/year.
"""

name = "6h_WeeklyPivot_Direction_1dVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point: (H + L + C) / 3"""
    return (high + low + close) / 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for pivot point ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    # Calculate weekly pivot point
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    wk_pivot = calculate_weekly_pivot(wk_high, wk_low, wk_close)
    
    # Align weekly pivot to 6h timeframe
    wk_pivot_aligned = align_htf_to_ltf(prices, df_1w, wk_pivot)

    # Daily volume spike: current > 1.8x average of last 8 days
    vol_ma = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup
        if (np.isnan(wk_pivot_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above weekly pivot + volume spike
            if (close[i] > wk_pivot_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price below weekly pivot + volume spike
            elif (close[i] < wk_pivot_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below weekly pivot
            if close[i] < wk_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above weekly pivot
            if close[i] > wk_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals