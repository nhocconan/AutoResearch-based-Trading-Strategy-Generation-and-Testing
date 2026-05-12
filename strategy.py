#!/usr/bin/env python3
# 4h_PriceAction_Reversal_1dTrend
# Hypothesis: Fade extreme price rejections at daily support/resistance with volume confirmation.
# Long when price rejects daily support (low touches/breaks below daily low then closes back above) with rising volume.
# Short when price rejects daily resistance (high touches/breaks above daily high then closes back below) with rising volume.
# Uses 1-day pivot points for support/resistance levels. Works in both bull and bear markets by fading reversals at key levels.
# Targets 20-30 trades/year to minimize fee drag.

name = "4h_PriceAction_Reversal_1dTrend"
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
    
    # Calculate 1-day pivot points (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for pivot calculation
    prev_high = df_1d['high'].shift(1).values  # Shift to use prior day only
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot, support, resistance
    pivot = (prev_high + prev_low + prev_close) / 3.0
    support1 = (2 * pivot) - prev_high
    resistance1 = (2 * pivot) - prev_low
    
    # Align to 4h timeframe (values update only at daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    support1_aligned = align_htf_to_ltf(prices, df_1d, support1)
    resistance1_aligned = align_htf_to_ltf(prices, df_1d, resistance1)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(support1_aligned[i]) or 
            np.isnan(resistance1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price rejects support (low touches/goes below support then closes back above)
            # with volume confirmation
            if low[i] <= support1_aligned[i] and close[i] > support1_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejects resistance (high touches/goes above resistance then closes back below)
            elif high[i] >= resistance1_aligned[i] and close[i] < resistance1_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back below support or reaches pivot
            if close[i] < support1_aligned[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back above resistance or reaches pivot
            if close[i] > resistance1_aligned[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals