#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R1S1_Breakout_Trend_Filter
# Hypothesis: Price reacts to weekly Camarilla pivot levels (R1/S1) derived from weekly timeframe. 
# Go long when price breaks above R1 with weekly uptrend (price > EMA200). 
# Go short when price breaks below S1 with weekly downtrend (price < EMA200).
# Weekly timeframe reduces noise and false breakouts, suitable for daily timeframe strategy.
# Weekly trend filter ensures alignment with higher timeframe momentum, improving win rate.
# Target: 10-25 trades/year per symbol to minimize fee drag.

name = "1d_Weekly_Camarilla_R1S1_Breakout_Trend_Filter"
timeframe = "1d"
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

    # Get weekly data for Camarilla pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels (R1, S1) from previous weekly bar
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    camarilla_width = (high_1w - low_1w) * 1.1 / 12
    r1 = close_1w + camarilla_width
    s1 = close_1w - camarilla_width
    
    # Weekly trend: EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + weekly uptrend (price > EMA200)
            if close[i] > r1_aligned[i] and close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + weekly downtrend (price < EMA200)
            elif close[i] < s1_aligned[i] and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal (price < EMA200)
            if close[i] < s1_aligned[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal (price > EMA200)
            if close[i] > r1_aligned[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals