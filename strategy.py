# The strategy uses 6h timeframe with weekly pivot points from 1w data and 1d EMA200 trend filter.
# Weekly pivot points are calculated using the standard formula (H+L+C)/3 for pivot, 
# with R1 = P + (H-L), S1 = P - (H-L), R2 = P + 2*(H-L), S2 = P - 2*(H-L).
# Entry conditions: 
#   Long: Price crosses above weekly R1 with price above 1d EMA200 (uptrend filter)
#   Short: Price crosses below weekly S1 with price below 1d EMA200 (downtrend filter)
# Exit conditions: 
#   Long: Price closes below weekly pivot point
#   Short: Price closes above weekly pivot point
# Position sizing: 0.25 (25% of capital) to manage risk and reduce drawdown.
# The weekly pivot provides strong support/resistance levels that work in both trending and ranging markets.
# The 1d EMA200 filter ensures we only trade in the direction of the higher timeframe trend.
# This combination should reduce false breakouts and improve win rate in both bull and bear markets.

#!/usr/bin/env python3
name = "6h_1w_Pivot_R1S1_Breakout_1dEMA200_Trend"
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
    volume = prices['volume'].values

    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly pivot points
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (High - Low)
    # S1 = Pivot - (High - Low)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = pivot_1w + (high_1w - low_1w)
    s1_1w = pivot_1w - (high_1w - low_1w)

    # Align weekly pivot points to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)

    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above weekly R1 with uptrend filter (price > EMA200)
            if close[i] > r1_aligned[i] and close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below weekly S1 with downtrend filter (price < EMA200)
            elif close[i] < s1_aligned[i] and close[i] < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly pivot (trend reversal signal)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly pivot (trend reversal signal)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals