#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_ChopFilter_v1
# Hypothesis: Use KAMA trend direction from 4h combined with RSI(14) for momentum and Choppiness Index for regime filtering.
# Enter long when KAMA trending up, RSI > 50, and choppy market (CHOP > 61.8); short when KAMA trending down, RSI < 50, and choppy.
# Designed to capture mean-reversion in ranging markets while avoiding strong trends. Works in both bull and bear markets by adapting to regime.

name = "4h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "4h"
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

    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # needs correction: should be rolling sum
    # Correct way: calculate ER using rolling window
    close_series = pd.Series(close)
    change = close_series.diff().abs()
    volatility = close_series.diff().abs().rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    # Avoid division by zero
    er = er.fillna(0).values

    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 0  # first value undefined

    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined

    # Choppiness Index (CHOP) - calculate on 4h data directly
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))))
    atr = atr.rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(14)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when range is zero

    # Conditions
    kama_up = kama_dir == 1
    kama_down = kama_dir == -1
    rsi_over_50 = rsi > 50
    rsi_under_50 = rsi < 50
    choppy = chop > 61.8  # ranging market

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # start after CHOP/RSI warmup
        if position == 0:
            # LONG: KAMA up, RSI > 50, choppy market
            if kama_up[i] and rsi_over_50[i] and choppy[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, RSI < 50, choppy market
            elif kama_down[i] and rsi_under_50[i] and choppy[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI < 50 or market trends (CHOP < 38.2)
            if not kama_up[i] or not rsi_over_50[i] or not choppy[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI > 50 or market trends (CHOP < 38.2)
            if not kama_down[i] or not rsi_under_50[i] or not choppy[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals