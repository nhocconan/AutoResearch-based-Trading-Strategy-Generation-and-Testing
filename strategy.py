#!/usr/bin/env python3
# 1h_4H_Trend_Signal_1D_Volume_Confirmation
# Hypothesis: The 4h trend direction filters 1h entries to avoid counter-trend trades.
# Long when 4h trend is up (close > EMA50) and 1h closes above the 1h EMA20 with volume confirmation.
# Short when 4h trend is down (close < EMA50) and 1h closes below the 1h EMA20 with volume confirmation.
# This reduces false breakouts and improves win rate in both bull and bear markets by aligning with higher timeframe momentum.
# Volume confirmation ensures entries occur with conviction, reducing whipsaws.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).

name = "1h_4H_Trend_Signal_1D_Volume_Confirmation"
timeframe = "1h"
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

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)

    # 4h EMA50 trend filter (needs only the completed 4h candle)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # 1h EMA20 for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(volume_ok[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 4h EMA50
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]

        if position == 0:
            # LONG: 4h trend up, price above 1h EMA20, volume confirmation
            if trend_up and close[i] > ema_20[i] and volume_ok[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: 4h trend down, price below 1h EMA20, volume confirmation
            elif trend_down and close[i] < ema_20[i] and volume_ok[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h trend turns down or price crosses below 1h EMA20
            if not trend_up or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h trend turns up or price crosses above 1h EMA20
            if not trend_down or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals