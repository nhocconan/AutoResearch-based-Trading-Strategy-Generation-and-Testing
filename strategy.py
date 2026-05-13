#!/usr/bin/env python3
"""
6h_ExponentialTrend_Pullback
Hypothesis: In trending markets (identified by 1d EMA alignment), 
price pulls back to the 6-day exponential moving average (EMA) on 6h chart 
provide high-probability entries in the direction of the daily trend. 
Volume confirmation filters out weak pulls. Works in both bull and bear 
by only trading with the dominant daily trend. Target: 20-30 trades/year.
"""

name = "6h_ExponentialTrend_Pullback"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Daily trend: 20 EMA and 50 EMA alignment
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = ema20_1d > ema50_1d
    daily_downtrend = ema20_1d < ema50_1d

    # 6h EMA: 6-period (approx 1 day of 6h bars)
    ema6_6h = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values

    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)

    # Align daily trend to 6h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values
        uptrend = daily_uptrend_aligned[i]
        downtrend = daily_downtrend_aligned[i]
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Daily uptrend + price at or near 6h EMA + volume spike
            if uptrend and close[i] <= ema6_6h[i] * 1.005 and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Daily downtrend + price at or near 6h EMA + volume spike
            elif downtrend and close[i] >= ema6_6h[i] * 0.995 and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above 6h EMA or trend changes
            if close[i] >= ema6_6h[i] * 1.01 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below 6h EMA or trend changes
            if close[i] <= ema6_6h[i] * 0.99 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals