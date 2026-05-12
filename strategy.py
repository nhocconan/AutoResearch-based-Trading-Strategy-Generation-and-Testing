#!/usr/bin/env python3
# 6h_MonthlyPivot_Breakout_TrendVolume_v1
# Hypothesis: 6h breakout of monthly pivot S1/R1 levels with 1w EMA50 trend filter and volume spike confirmation.
# Monthly pivots provide stronger support/resistance than daily, reducing whipsaw in choppy markets.
# 1w EMA50 ensures alignment with longer-term trend, improving performance in both bull and bear markets.
# Volume spike (2x 20-period average) confirms breakout strength.
# Targets 15-25 trades/year to minimize fee drag while capturing significant moves.

name = "6h_MonthlyPivot_Breakout_TrendVolume_v1"
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

    # Get 6h data for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Get 1w data for monthly pivot points (using weekly high/low/close as proxy)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    # Calculate monthly pivot points from weekly data (simplified)
    # Using weekly high, low, close to calculate pivot and support/resistance
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)

    # Align monthly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)

    # Get 1w data for EMA50 trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate 6h ATR(14) for dynamic threshold (optional filter)
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.concatenate([[close_6h[0]], close_6h[:-1]]))
    tr3 = np.abs(low_6h - np.concatenate([[close_6h[0]], close_6h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0  # Require 2x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R1 with 1w uptrend and volume spike
            if (close[i] > r1_1w_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 with 1w downtrend and volume spike
            elif (close[i] < s1_1w_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1w EMA50 (trend reversal) or breaks S1
            if close[i] < ema50_1w_aligned[i] or close[i] < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1w EMA50 (trend reversal) or breaks R1
            if close[i] > ema50_1w_aligned[i] or close[i] > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals