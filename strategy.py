#!/usr/bin/env python3
# 12h_RSI_Divergence_Volume
# Hypothesis: On 12h timeframe, identify RSI divergences with price (bullish/bearish) on 1d RSI.
# Enter long when price makes lower low but RSI makes higher low (bullish divergence) with volume confirmation.
# Enter short when price makes higher high but RSI makes lower high (bearish divergence) with volume confirmation.
# Exit when RSI crosses above 70 (long) or below 30 (short) to capture mean reversion.
# Uses 1d RSI to avoid noise, with volume filter to ensure momentum behind moves.
# Designed for fewer trades (<30/year) to minimize fee drag while capturing high-probability reversals.

name = "12h_RSI_Divergence_Volume"
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

    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate 14-period RSI on daily closes
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    # For first 13 values, RSI is undefined (set to 50 as neutral)
    rsi_1d[:13] = 50

    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i < 19:
            vol_avg_20[i] = np.nan
        else:
            vol_avg_20[i] = vol_sum / 20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Need at least 2 days of data to check divergence
        if i < 2 * 2:  # Need at least 2 aligned 1d bars (each 1d = 2 bars of 12h)
            signals[i] = 0.0
            continue

        # Get current and previous 1d bar indices in 12h timeframe
        # Each 1d = 2 bars of 12h
        curr_1d_idx = i // 2
        prev_1d_idx = curr_1d_idx - 1

        # Map 12h bar index to approximate 1d bar boundaries
        # For simplicity, use aligned RSI values directly (already synchronized)
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-2] and  # Lower low in price (approx 1d lookback)
                rsi_1d_aligned[i] > rsi_1d_aligned[i-2] and  # Higher low in RSI
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (high[i] > high[i-2] and  # Higher high in price
                  rsi_1d_aligned[i] < rsi_1d_aligned[i-2] and  # Lower high in RSI
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when RSI crosses above 70 (overbought)
            if rsi_1d_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when RSI crosses below 30 (oversold)
            if rsi_1d_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals