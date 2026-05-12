#!/usr/bin/env python3
# 1d_KAMA_Trend_Follow_With_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing smooth trend signals. Combined with a Chop filter (range-bound detection) to avoid whipsaws in sideways markets.
# Long when KAMA slopes up and Chop < 61.8 (trending market). Short when KAMA slopes down and Chop < 61.8.
# Exit on opposite KAMA slope or when Chop > 61.8 (range market). Designed for low trade frequency on daily timeframe.
# Works in bull/bear by following adaptive trend and avoiding range-bound chop.

name = "1d_KAMA_Trend_Follow_With_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=er_length, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_chop(high, low, close, window=14):
    """Choppiness Index: higher values indicate ranging markets"""
    atr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum()
    highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
    lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
    range_max_min = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / range_max_min) / np.log10(window)
    return chop.fillna(50).values  # fill NaN with middle value

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Calculate KAMA (adaptive trend)
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    kama_slope = kama - np.roll(kama, 1)  # slope = current - previous

    # Calculate Chop (range filter)
    chop = calculate_chop(high, low, close, window=14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if KAMA slope or Chop is invalid
        if np.isnan(kama_slope[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade in trending markets (Chop < 61.8)
        if chop[i] < 61.8:
            if position == 0:
                # ENTER LONG: KAMA sloping up
                if kama_slope[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # ENTER SHORT: KAMA sloping down
                elif kama_slope[i] < 0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: KAMA slopes down or Chop > 61.8 (range)
                if kama_slope[i] < 0 or chop[i] >= 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: KAMA slopes up or Chop > 61.8 (range)
                if kama_slope[i] > 0 or chop[i] >= 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging markets, flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0

    return signals