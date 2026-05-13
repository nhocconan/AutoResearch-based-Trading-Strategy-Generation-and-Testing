#!/usr/bin/env python3
# 1d_Williams_Alligator_1wTrend_Filter
# Hypothesis: Use Williams Alligator (Jaw, Teeth, Lips) on 1d to detect trend direction and entry signals.
# Go long when Lips cross above Teeth and both are above Jaw (bullish alignment) with 1w uptrend filter.
# Go short when Lips cross below Teeth and both are below Jaw (bearish alignment) with 1w downtrend filter.
# The Alligator helps identify trending vs ranging markets; we only trade when aligned with higher timeframe trend.
# Target: 15-25 trades/year per symbol to minimize fee drag.

name = "1d_Williams_Alligator_1wTrend_Filter"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Alligator on 1d: SMoothed Moving Average (SMMA)
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(arr, period):
        # Smoothed Moving Average: similar to EMA but with alpha = 1/period
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result

    jaw = smma(close, 13)  # SMMA(13)
    teeth = smma(close, 8)  # SMMA(8)
    lips = smma(close, 5)   # SMMA(5)

    # 1w trend: EMA50
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w trend to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after Alligator warmup
        # Skip if any required value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) + 1w uptrend
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) + 1w downtrend
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish alignment (Lips < Teeth < Jaw) or trend reversal
            if lips[i] < teeth[i] and teeth[i] < jaw[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish alignment (Lips > Teeth > Jaw) or trend reversal
            if lips[i] > teeth[i] and teeth[i] > jaw[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals