#!/usr/bin/env python3
# 6h_MACD_Trend_Filter
# Hypothesis: MACD(12,26,9) on 6m timeframe captures medium-term momentum; confirmed by 1w trend filter (price > 200-period EMA) and volume spikes (>1.5x 50-period average). Enter long when MACD line crosses above signal line and price > 1w EMA200 with volume spike; short when MACD line crosses below signal line and price < 1w EMA200 with volume spike. Exit on MACD reverse crossover. Targets 15-35 trades/year to minimize fee drag and work in both bull/bear markets via trend filter.

name = "6h_MACD_Trend_Filter"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate MACD on 6h
    fast_period = 12
    slow_period = 26
    signal_period = 9

    ema_fast = pd.Series(close).ewm(span=fast_period, adjust=False, min_periods=fast_period).mean().values
    ema_slow = pd.Series(close).ewm(span=slow_period, adjust=False, min_periods=slow_period).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal_period, adjust=False, min_periods=signal_period).mean().values

    # 1w EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Volume confirmation: volume > 1.5x 50-period average
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(max(fast_period, slow_period, signal_period), n):
        # Skip if any required value is NaN
        if (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_avg_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: MACD line crosses above signal line + price > 1w EMA200 + volume spike
            if (macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1] and
                close[i] > ema200_1w_aligned[i] and
                volume[i] > vol_avg_50[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: MACD line crosses below signal line + price < 1w EMA200 + volume spike
            elif (macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1] and
                  close[i] < ema200_1w_aligned[i] and
                  volume[i] > vol_avg_50[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: MACD line crosses below signal line
            if macd_line[i] < signal_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: MACD line crosses above signal line
            if macd_line[i] > signal_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals