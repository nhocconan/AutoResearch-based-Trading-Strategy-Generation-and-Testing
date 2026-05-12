#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Volume_Confirmation
Hypothesis: On 12h timeframe, KAMA (Kaufman Adaptive Moving Average) adapts to market noise, 
providing a reliable trend signal. Long when price > KAMA with volume > 1.5x 20-period average, 
short when price < KAMA with volume surge. Uses 1d Bollinger Band width < 50th percentile to 
filter choppy regimes, ensuring trades occur in trending markets. Targets 12-37 trades/year 
(50-150 total over 4 years) with low turnover to minimize fee flood. Works in bull via 
momentum continuation and bear via mean-reversion at extremes with trend filter.
"""

name = "12h_KAMA_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d KAMA for trend
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama_1d = calculate_kama(close_1d, length=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # Calculate 1d Bollinger Band width (20, 2) for squeeze filter
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma20_1d
    # Percentile rank of bb_width over lookback
    bb_width_rank = pd.Series(bb_width_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_1d, bb_width_rank)

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current 12h bar
        kama = kama_1d_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(kama) or np.isnan(bb_rank) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Squeeze filter: only trade when BB width is in lower 50% (contraction)
        if bb_rank > 0.5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + volume surge
            if (close[i] > kama and volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + volume surge
            elif (close[i] < kama and volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA
            if close[i] < kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA
            if close[i] > kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals