#!/usr/bin/env python3
# 4h_Stochastic_Reversal_TrendFilter
# Hypothesis: 4h Stochastic oscillator identifies overbought/oversold conditions for mean reversion.
# Uses 1d EMA as trend filter to trade only in direction of higher timeframe trend.
# In uptrend: buy when Stochastic crosses above 20 from below (oversold bounce).
# In downtrend: sell when Stochastic crosses below 80 from above (overbought rejection).
# Designed for 20-50 trades/year with clear entry/exit rules to minimize fee drag.
# Works in bull/bear markets by following 1d trend direction.

name = "4h_Stochastic_Reversal_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get 4h data for price action and Stochastic calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Calculate Stochastic Oscillator (14,3,3)
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    k_percent = np.where((highest_high - lowest_low) != 0,
                         ((close_4h - lowest_low) / (highest_high - lowest_low)) * 100,
                         50)  # Avoid division by zero, set to neutral 50
    # %D = 3-period SMA of %K
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values

    # Align Stochastic and EMA to 4h timeframe
    k_percent_aligned = align_htf_to_ltf(prices, df_4h, k_percent)
    d_percent_aligned = align_htf_to_ltf(prices, df_4h, d_percent)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(k_percent_aligned[i]) or np.isnan(d_percent_aligned[i]) or
            np.isnan(ema20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: In uptrend (price > 1d EMA20), Stochastic crosses above 20 from below
            if (close[i] > ema20_1d_aligned[i] and
                k_percent_aligned[i] > 20 and d_percent_aligned[i-1] <= 20 and
                k_percent_aligned[i-1] <= 20):
                signals[i] = 0.25
                position = 1
            # SHORT: In downtrend (price < 1d EMA20), Stochastic crosses below 80 from above
            elif (close[i] < ema20_1d_aligned[i] and
                  k_percent_aligned[i] < 80 and d_percent_aligned[i-1] >= 80 and
                  k_percent_aligned[i-1] >= 80):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Stochastic crosses below 80 (overbought) or trend reversal
            if (k_percent_aligned[i] < 80 and d_percent_aligned[i-1] >= 80) or \
               (close[i] < ema20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Stochastic crosses above 20 (oversold) or trend reversal
            if (k_percent_aligned[i] > 20 and d_percent_aligned[i-1] <= 20) or \
               (close[i] > ema20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals