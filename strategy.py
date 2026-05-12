#!/usr/bin/env python3
"""
1d_VWAP_Range_Reversal_1wTrend
Hypothesis: Price reverting to 1w VWAP from daily extremes works in both bull and bear markets. Uses 1w VWAP as dynamic mean, with 1d high/low as reversal zones. Trend filter ensures we only take reversals in direction of weekly trend, avoiding counter-trend whipsaws. Low-frequency signals reduce fee drag.
"""

name = "1d_VWAP_Range_Reversal_1wTrend"
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
    volume = prices['volume'].values

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate 1w VWAP
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_numerator = (typical_price * df_1w['volume']).cumsum()
    vwap_denominator = df_1w['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    vwap = vwap.replace([np.inf, -np.inf], np.nan).ffill().bfill().values

    # Align 1w VWAP to daily
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)

    # 1d high/low for reversal zones
    daily_high = high
    daily_low = low

    # 1w trend: EMA50
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at daily low + below VWAP + weekly uptrend
            if (low[i] <= daily_low[i] and 
                close[i] < vwap_aligned[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at daily high + above VWAP + weekly downtrend
            elif (high[i] >= daily_high[i] and 
                  close[i] > vwap_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above VWAP
            if close[i] > vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below VWAP
            if close[i] < vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals