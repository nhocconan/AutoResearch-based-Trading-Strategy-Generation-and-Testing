#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_Trend_Filter
# Hypothesis: Enter long when price breaks above weekly R4 pivot level in the direction of daily EMA200 trend, confirmed by volume spike.
# Enter short when price breaks below weekly S4 pivot level in the direction of daily EMA200 trend, confirmed by volume spike.
# Weekly pivots define key support/resistance levels. Breaks of R4/S4 indicate strong momentum in the direction of the break.
# Trend filter ensures alignment with higher timeframe momentum, reducing false breakouts.
# Volume spike confirms institutional participation. Works in bull (breakouts above R4 in uptrend) and bear (breakdowns below S4 in downtrend).
# Low frequency due to weekly pivot requirement and strict volume confirmation.

name = "6h_Weekly_Pivot_Breakout_Trend_Filter"
timeframe = "6h"
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

    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot points calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    r4 = r3 + (high_1w - low_1w)
    s4 = s3 - (high_1w - low_1w)
    
    # Daily trend: EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly pivots and daily EMA to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume spike: volume > 2.0 * 4-period average (1 day worth at 6h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly R4 + daily uptrend + volume spike
            if close[i] > r4_aligned[i] and close[i] > ema200_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly S4 + daily downtrend + volume spike
            elif close[i] < s4_aligned[i] and close[i] < ema200_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly pivot OR trend reversal
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
            if close[i] < pivot_aligned[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly pivot OR trend reversal
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
            if close[i] > pivot_aligned[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals