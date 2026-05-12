#!/usr/bin/env python3
"""
6h_RSI_Divergence_Pivot_Reversal
Hypothesis: On 6h timeframe, use daily RSI divergence (bullish/bearish) at 1d pivot levels (S1/S2/R1/R2) for mean-reversion entries in both bull and bear markets. Divergence signals exhaustion, pivots provide support/resistance, and 6h timeframe avoids excessive trading. Target 15-35 trades/year with disciplined exits.
"""

name = "6h_RSI_Divergence_Pivot_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_pivots(high, low, close):
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for RSI and pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d RSI
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Calculate 1d pivots (using prior day's HLC)
    pivot, r1, r2, s1, s2 = calculate_pivots(high_1d, low_1d, close_1d)
    # Align pivot levels (use prior day's values for current day)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)

    # RSI divergence detection: look for price making new high/low while RSI does not
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    lookback = 6  # 6 periods (~1.5 days on 6h chart)
    price_low_6 = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    price_high_6 = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    rsi_low_6 = pd.Series(rsi_1d_aligned).rolling(window=lookback, min_periods=lookback).min().values
    rsi_high_6 = pd.Series(rsi_1d_aligned).rolling(window=lookback, min_periods=lookback).max().values

    bullish_div = (low == price_low_6) & (rsi_1d_aligned > rsi_low_6)
    bearish_div = (high == price_high_6) & (rsi_1d_aligned < rsi_high_6)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish RSI divergence at or near S1/S2 support
            if bullish_div[i] and (close[i] <= s1_aligned[i] * 1.005 or close[i] <= s2_aligned[i] * 1.005):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish RSI divergence at or near R1/R2 resistance
            elif bearish_div[i] and (close[i] >= r1_aligned[i] * 0.995 or close[i] >= r2_aligned[i] * 0.995):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot or RSI shows bearish divergence
            if close[i] >= pivot_aligned[i] or bearish_div[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or RSI shows bullish divergence
            if close[i] <= pivot_aligned[i] or bullish_div[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals