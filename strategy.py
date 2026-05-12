#!/usr/bin/env python3
# 1d_Donchian_Breakout_1wTrend_Volume_Confirmation
# Hypothesis: Daily Donchian breakouts (20-period) aligned with weekly trend (EMA50) and volume confirmation capture strong directional moves.
# Weekly trend filter prevents counter-trend trades, reducing false breakouts in ranging or counter-trend markets.
# Volume surge confirms institutional participation. Works in bull/bear by following weekly trend.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "1d_Donchian_Breakout_1wTrend_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Donchian Channel (20) on daily
    # Upper = max(high, 20)
    # Lower = min(low, 20)
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest[i] = np.max(high[i - lookback + 1:i + 1])
        lowest[i] = np.min(low[i - lookback + 1:i + 1])

    # Volume confirmation: current > 1.8x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup and Donchian lookback
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest[i]) or 
            np.isnan(lowest[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian Upper + weekly UPTREND + volume confirmation
            if (close[i] > highest[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian Lower + weekly DOWNTREND + volume confirmation
            elif (close[i] < lowest[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian Lower (breakdown of structure)
            if close[i] < lowest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian Upper (breakout of structure)
            if close[i] > highest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals