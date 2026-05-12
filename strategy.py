#!/usr/bin/env python3
# 12h_TripleConfirmation_Breakout
# Hypothesis: Use 1-week Donchian channel breakout combined with 1-day RSI momentum and volume confirmation to capture major trend moves.
# The weekly timeframe filters out noise, while daily RSI and volume provide entry timing.
# Designed to work in both bull and bear markets by requiring alignment across multiple timeframes.
# Target: 15-30 trades per year to minimize fee drag.

name = "12h_TripleConfirmation_Breakout"
timeframe = "12h"
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

    # Get weekly data for Donchian channel (primary trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Get daily data for RSI and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate weekly Donchian channels (20-period)
    # Using expanding window approach with min_periods for proper initialization
    high_max_20 = np.full(len(high_1w), np.nan)
    low_min_20 = np.full(len(low_1w), np.nan)
    
    for i in range(len(high_1w)):
        if i >= 19:  # 20 periods minimum
            high_max_20[i] = np.max(high_1w[i-19:i+1])
            low_min_20[i] = np.min(low_1w[i-19:i+1])
        elif i == 0:
            high_max_20[i] = high_1w[0]
            low_min_20[i] = low_1w[0]

    # Calculate daily RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 gains
    avg_loss[13] = np.mean(loss[1:14])  # First average of first 14 losses
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    # Handle division by zero when avg_loss is 0
    rsi_1d = np.where(avg_loss == 0, 100, rsi_1d)
    # Handle case where both gain and loss are 0 (shouldn't happen in practice but safe)
    rsi_1d = np.where((avg_gain == 0) & (avg_loss == 0), 50, rsi_1d)

    # Calculate daily volume average (20-period)
    volume_avg_20d = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            volume_avg_20d[i] = np.mean(volume_1d[i-19:i+1])

    # Align HTF indicators to 12h timeframe
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    volume_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, volume_avg_20d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient data for indicators
        # Skip if any required value is NaN
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_avg_20d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly Donchian high + RSI > 50 (bullish momentum) + volume above average
            if (close[i] > donchian_high_1w_aligned[i] and
                rsi_1d_aligned[i] > 50 and
                volume[i] > volume_avg_20d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low + RSI < 50 (bearish momentum) + volume above average
            elif (close[i] < donchian_low_1w_aligned[i] and
                  rsi_1d_aligned[i] < 50 and
                  volume[i] > volume_avg_20d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low (trend reversal)
            if close[i] < donchian_low_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high (trend reversal)
            if close[i] > donchian_high_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals