#!/usr/bin/env python3
# 6h_RSI_Divergence_Trend_Reversal
# Hypothesis: Combine RSI divergence with trend confirmation on higher timeframe for reversals.
# Bullish divergence (price makes lower low, RSI makes higher low) + price above 1w EMA200 = long.
# Bearish divergence (price makes higher high, RSI makes lower high) + price below 1w EMA200 = short.
# Uses 60-period RSI to reduce noise, weekly EMA200 for trend filter.
# Designed for low frequency (20-60 trades/year) with high conviction signals.

name = "6h_RSI_Divergence_Trend_Reversal"
timeframe = "6h"
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

    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)

    # Calculate RSI(60) on 6h closes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/60, adjust=False, min_periods=60).mean()
    avg_loss = loss.ewm(alpha=1/60, adjust=False, min_periods=60).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined

    # Find peaks and troughs for divergence detection
    # Use 5-bar windows for swing points
    def find_swing_points(arr, window=2):
        """Find local minima and maxima"""
        n = len(arr)
        mins = np.zeros(n, dtype=bool)
        maxs = np.zeros(n, dtype=bool)
        for i in range(window, n - window):
            if arr[i] == np.min(arr[i-window:i+window+1]):
                mins[i] = True
            if arr[i] == np.max(arr[i-window:i+window+1]):
                maxs[i] = True
        return mins, maxs

    price_mins, price_maxs = find_swing_points(close, 2)
    rsi_mins, rsi_maxs = find_swing_points(rsi, 2)

    # Track recent swing points for divergence
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)

    # Store last swing points
    last_price_min = np.nan
    last_price_max = np.nan
    last_rsi_min = np.nan
    last_rsi_max = np.nan

    for i in range(10, n):
        # Update swing points
        if price_mins[i]:
            last_price_min = low[i]
            last_rsi_min = rsi[i]
        if price_maxs[i]:
            last_price_max = high[i]
            last_rsi_max = rsi[i]

        # Check for bullish divergence: lower low in price, higher low in RSI
        if (not np.isnan(last_price_min) and price_mins[i] and 
            low[i] < last_price_min and rsi[i] > last_rsi_min):
            bullish_div[i] = True
            # Reset tracking after signal
            last_price_min = np.nan
            last_rsi_min = np.nan

        # Check for bearish divergence: higher high in price, lower high in RSI
        if (not np.isnan(last_price_max) and price_maxs[i] and 
            high[i] > last_price_max and rsi[i] < last_rsi_max):
            bearish_div[i] = True
            # Reset tracking after signal
            last_price_max = np.nan
            last_rsi_max = np.nan

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Wait for RSI to stabilize
        # Skip if any required value is NaN
        if np.isnan(ema_200_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish divergence + price above weekly EMA200 (uptrend bias)
            if bullish_div[i] and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + price below weekly EMA200 (downtrend bias)
            elif bearish_div[i] and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence or price breaks below EMA200
            if bearish_div[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence or price breaks above EMA200
            if bullish_div[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals