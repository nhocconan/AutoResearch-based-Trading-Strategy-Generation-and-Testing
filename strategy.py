#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Width_Range_MeanReversion
# Hypothesis: Use 1d Camarilla pivot width as range detector; mean-revert at S1/R1 levels when range is narrow (low volatility).
# In narrow ranges (width < 20-period average), buy at S1, sell at R1 with 1d trend filter.
# Works in both bull/bear by capturing mean reversion in consolidation phases.

name = "12h_Camarilla_Pivot_Width_Range_MeanReversion"
timeframe = "12h"
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

    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels: R1, S1, and PP (pivot point)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pp_1d = typical_price.values
    hl_range = df_1d['high'] - df_1d['low']
    r1_1d = df_1d['close'].values + hl_range.values * 1.1 / 4
    s1_1d = df_1d['close'].values - hl_range.values * 1.1 / 4
    
    # Align 1d Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)

    # Get 1d data for EMA trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Pivot width (R1-S1) for range detection
    pivot_width_1d = r1_1d - s1_1d
    pivot_width_avg = pd.Series(pivot_width_1d).rolling(window=20, min_periods=20).mean().values
    pivot_width_avg_aligned = align_htf_to_ltf(prices, df_1d, pivot_width_avg)

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pivot_width_avg_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Range condition: current pivot width < average width (narrow range = mean revert opportunity)
        is_narrow_range = pivot_width_1d[i] < pivot_width_avg_aligned[i]

        if position == 0:
            # LONG: Price at S1 + narrow range + price above 1d EMA50 (uptrend bias)
            if (close[i] <= s1_1d_aligned[i] * 1.005 and  # Allow small buffer
                is_narrow_range and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R1 + narrow range + price below 1d EMA50 (downtrend bias)
            elif (close[i] >= r1_1d_aligned[i] * 0.995 and  # Allow small buffer
                  is_narrow_range and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches PP or volatility expands (width > average)
            if (close[i] >= pp_1d_aligned[i] * 0.995 or 
                pivot_width_1d[i] > pivot_width_avg_aligned[i] * 1.1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches PP or volatility expands
            if (close[i] <= pp_1d_aligned[i] * 1.005 or 
                pivot_width_1d[i] > pivot_width_avg_aligned[i] * 1.1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals