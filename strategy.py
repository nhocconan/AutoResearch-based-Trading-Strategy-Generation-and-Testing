#!/usr/bin/env python3
# 6h_RSI_Divergence_Volume_Trend
# Hypothesis: Combining RSI divergence with volume confirmation and higher-timeframe trend filters
# to capture high-probability reversal entries in both bull and bear markets. The 6h timeframe
# reduces noise and limits trade frequency to avoid fee drag, while the daily trend filter ensures
# alignment with the dominant market direction.
# Strategy:
#   - Regular bullish divergence: price makes lower low, RSI makes higher low
#   - Regular bearish divergence: price makes higher high, RSI makes lower high
#   - Confirm with volume spike (>1.5x 20-period average)
#   - Use daily EMA50 as trend filter: only long when price > daily EMA50, short when price < daily EMA50
#   - Exit on opposite signal or when price crosses 20-period EMA (mean reversion)
# Target: 20-40 trades per year on 6h to stay within optimal range.

name = "6h_RSI_Divergence_Volume_Trend"
timeframe = "6h"
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

    # Calculate RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # EMA20 for exit signal
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Get daily EMA50 for trend filter (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema20[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Check for bullish divergence: price lower low, RSI higher low
            bull_div = False
            if i >= 2:
                # Look for recent swing low in price and RSI
                if low[i] < low[i-1] and rsi[i] > rsi[i-1]:
                    bull_div = True
                # Also check 2-bar lookback for more robustness
                elif low[i] < low[i-2] and rsi[i] > rsi[i-2]:
                    bull_div = True

            # Check for bearish divergence: price higher high, RSI lower high
            bear_div = False
            if i >= 2:
                if high[i] > high[i-1] and rsi[i] < rsi[i-1]:
                    bear_div = True
                elif high[i] > high[i-2] and rsi[i] < rsi[i-2]:
                    bear_div = True

            # Volume confirmation
            vol_spike = volume[i] > vol_avg_20[i] * 1.5

            # LONG: Bullish divergence + volume spike + price above daily EMA50
            if bull_div and vol_spike and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + volume spike + price below daily EMA50
            elif bear_div and vol_spike and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA20 (mean reversion)
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA20 (mean reversion)
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals