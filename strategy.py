#!/usr/bin/env python3
# 4h_Flipping_Backtester_v1
# Hypothesis: Combine 1-day Williams %R with 4-hour RSI for mean-reversion entries in both bull and bear markets.
# Williams %R identifies overbought/oversold conditions on the daily chart, while RSI on 4h provides timing.
# Trades only when both timeframes agree on extreme conditions, reducing false signals.
# Uses volume confirmation to ensure institutional participation.
# Designed for 15-35 trades/year per symbol, works in both bull and bear via mean-reversion logic.

name = "4h_Flipping_Backtester_v1"
timeframe = "4h"
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

    # Get daily data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)

    # Williams %R on daily: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero

    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)

    # RSI on 4h (14-period)
    delta = pd.Series(close).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: current volume > 1.3x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Williams %R conditions: oversold (< -80) or overbought (> -20)
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20

        if position == 0:
            # LONG: Daily oversold AND 4h RSI rising from oversold AND volume
            if oversold and rsi[i] < 30 and rsi[i] > rsi[i-1] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Daily overbought AND 4h RSI falling from overbought AND volume
            elif overbought and rsi[i] > 70 and rsi[i] < rsi[i-1] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI reaches overbought OR Williams %R exits oversold
            if rsi[i] >= 70 or williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI reaches oversold OR Williams %R exits overbought
            if rsi[i] <= 30 or williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals