#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Filter
Hypothesis: On 12h timeframe, use Kaufman Adaptive Moving Average (KAMA) to determine trend direction, filter entries with RSI to avoid overextended moves, and use volume confirmation to ensure momentum. The strategy takes long positions when KAMA is rising and RSI is between 50-70 (bullish but not overbought), and short positions when KAMA is falling and RSI is between 30-50 (bearish but not oversold). Volume must be above average to confirm the move. Designed for low trade frequency to minimize fee drag and work in both bull and bear markets.
"""

name = "12h_KAMA_Direction_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get 1d data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on 1d
    # Parameters: fast=2, slow=30
    def kama(close, length=10, fast=2, slow=30):
        # Calculate Efficiency Ratio (ER)
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # For array, we need to compute per point
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if i == 0:
                er[i] = 0
            else:
                diff = np.abs(close[i] - close[i-length])
                vol = np.sum(np.abs(np.diff(close[i-length:i+1])))
                if vol != 0:
                    er[i] = diff / vol
                else:
                    er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals

    # Calculate RSI on 1d
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        # Wilder's smoothing
        avg_gain[length-1] = np.mean(gain[:length])
        avg_loss[length-1] = np.mean(loss[:length])
        for i in range(length, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals

    kama_1d = kama(close_1d, length=10, fast=2, slow=30)
    rsi_1d = rsi(close_1d, length=14)

    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.zeros_like(kama_1d)
    kama_dir[1:] = np.where(kama_1d[1:] > kama_1d[:-1], 1, np.where(kama_1d[1:] < kama_1d[:-1], -1, 0))

    # Align KAMA direction and RSI to 12h timeframe
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Volume confirmation: volume > 1.5x 24-period average (approx 12 hours)
    vol_avg_24 = np.zeros_like(volume)
    vol_series = pd.Series(volume)
    vol_avg_24 = vol_series.rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period for indicators
        # Skip if any required value is NaN
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (bullish trend) + RSI between 50-70 (bullish but not overbought) + volume confirmation
            if (kama_dir_aligned[i] == 1 and 
                50 <= rsi_1d_aligned[i] <= 70 and
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (bearish trend) + RSI between 30-50 (bearish but not oversold) + volume confirmation
            elif (kama_dir_aligned[i] == -1 and 
                  30 <= rsi_1d_aligned[i] <= 50 and
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down OR RSI becomes overbought (>70) OR volume drops
            if (kama_dir_aligned[i] == -1 or 
                rsi_1d_aligned[i] > 70 or
                volume[i] <= vol_avg_24[i] * 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up OR RSI becomes oversold (<30) OR volume drops
            if (kama_dir_aligned[i] == 1 or 
                rsi_1d_aligned[i] < 30 or
                volume[i] <= vol_avg_24[i] * 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals