#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_MeanReversion
# Hypothesis: On 1d timeframe, use KAMA trend filter with RSI mean reversion in ranging markets (Choppiness > 61.8).
# In ranging markets (high chop), buy when RSI oversold (<40) and price above KAMA, sell when RSI overbought (>60) and price below KAMA.
# This avoids strong trends where mean reversion fails and whipsaws in low chop.
# Designed for 10-30 trades per year to minimize fee drag. Works in bull/bear by adapting to regime.

name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Calculate ER (Efficiency Ratio) for KAMA
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of absolute changes
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # FAST=2, SLOW=30
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate RSI(14)
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # First 14 values are not valid due to min_periods
    rsi[:14] = np.nan

    # Calculate Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # True high/low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where(atr != 0, 100 * np.log10((max_high - min_low) / (atr * 14)) / np.log10(14), 50)
    chop[:14] = np.nan  # Not enough data

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required data is NaN
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade in ranging market (Choppiness > 61.8)
        if chop[i] > 61.8:
            if position == 0:
                # LONG: RSI oversold and price above KAMA (bullish mean reversion)
                if rsi[i] < 40 and close[i] > kama[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: RSI overbought and price below KAMA (bearish mean reversion)
                elif rsi[i] > 60 and close[i] < kama[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: RSI overbought or price below KAMA
                if rsi[i] > 60 or close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: RSI oversold or price above KAMA
                if rsi[i] < 40 or close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending market (low chop), do not trade to avoid whipsaws
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0

    return signals