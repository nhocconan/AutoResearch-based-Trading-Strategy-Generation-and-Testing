#!/usr/bin/env python3
# 6h_Aroon_Trend_Strength_1dTrend_Filter
# Hypothesis: Aroon Up/Down identifies trend strength and direction. Combined with 1-day trend filter (price above/below 200-day EMA) to avoid counter-trend trades. Works in bull/bear by following the higher timeframe trend direction. Uses 6h timeframe with daily EMA200 trend filter for higher timeframe context.

name = "6h_Aroon_Trend_Strength_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate Aroon (25-period)
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period, n):
        # Periods since highest high
        highest_high_idx = np.argmax(high[i-period+1:i+1]) + i - period + 1
        periods_since_high = i - highest_high_idx
        aroon_up[i] = ((period - 1 - periods_since_high) / (period - 1)) * 100
        
        # Periods since lowest low
        lowest_low_idx = np.argmin(low[i-period+1:i+1]) + i - period + 1
        periods_since_low = i - lowest_low_idx
        aroon_down[i] = ((period - 1 - periods_since_low) / (period - 1)) * 100

    # 1-day EMA200 trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):  # Start after EMA200 warmup
        if (np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Aroon Up > Aroon Down (uptrend) + price above 1d EMA200
            if (aroon_up[i] > aroon_down[i] and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Aroon Down > Aroon Up (downtrend) + price below 1d EMA200
            elif (aroon_down[i] > aroon_up[i] and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Aroon Down > Aroon Up (trend weakening)
            if aroon_down[i] > aroon_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Aroon Up > Aroon Down (trend weakening)
            if aroon_up[i] > aroon_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals