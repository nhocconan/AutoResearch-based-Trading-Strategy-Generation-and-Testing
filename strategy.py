#!/usr/bin/env python3
# 4h_RSI_Div_Volume_Reversal
# Hypothesis: 4-hour RSI divergence with volume confirmation captures reversals in both bull and bear markets.
# Uses RSI(14) to detect overbought/oversold conditions and price-volume divergence for reversal signals.
# Volume filter ensures trades occur with institutional participation, reducing false signals.
# Designed for 20-50 trades per year to minimize fee drag while maintaining edge in ranging/trending markets.

name = "4h_RSI_Div_Volume_Reversal"
timeframe = "4h"
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

    # Calculate RSI(14) on close prices
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Calculate volume SMA(20) for confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if required data is not ready
        if np.isnan(rsi[i]) or np.isnan(volume_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold (<30) with bullish divergence and volume confirmation
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (rsi[i] < 30 and 
                i >= 2 and 
                close[i] < close[i-2] and 
                rsi[i] > rsi[i-2] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) with bearish divergence and volume confirmation
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (rsi[i] > 70 and 
                  i >= 2 and 
                  close[i] > close[i-2] and 
                  rsi[i] < rsi[i-2] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral zone (40-60) or overbought
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral zone (40-60) or oversold
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals