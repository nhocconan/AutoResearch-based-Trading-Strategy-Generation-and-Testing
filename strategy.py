#!/usr/bin/env python3
"""
1d_SupportResistance_Reversal_WeeklyTrend
Hypothesis: Use weekly price extremes (4-week high/low) as dynamic support/resistance levels.
In trending markets, price respects these levels as pullback targets; in ranging markets,
they act as reversal zones. Enter on reversal from these levels with weekly trend filter.
Weekly trend filters out counter-trend trades, reducing whipsaws in strong trends.
Targets 15-25 trades/year by requiring confluence of level touch, reversal signal, and trend alignment.
"""

name = "1d_SupportResistance_Reversal_WeeklyTrend"
timeframe = "1d"
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

    # Get weekly data for trend filter and levels ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Calculate 4-week high and low (20 trading days) on weekly data
    high_4w = pd.Series(df_1w['high']).rolling(window=4, min_periods=4).max().values
    low_4w = pd.Series(df_1w['low']).rolling(window=4, min_periods=4).min().values
    
    # Weekly trend: price above/below 20-week EMA
    close_1w = df_1w['close'].values
    ema_20w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = close_1w > ema_20w
    weekly_downtrend = close_1w < ema_20w

    # Align weekly indicators to daily timeframe
    high_4w_aligned = align_htf_to_ltf(prices, df_1w, high_4w)
    low_4w_aligned = align_htf_to_ltf(prices, df_1w, low_4w)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)

    # Daily reversal signals: look for rejection at levels
    # Bullish rejection: long wick below support, close near high
    body_size = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    bullish_rejection = (lower_wick > 2 * body_size) & (close > open_)
    bearish_rejection = (upper_wick > 2 * body_size) & (close < open_)

    # Volume confirmation: current volume > 1.5x average of last 10 days
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup
        if (np.isnan(high_4w_aligned[i]) or np.isnan(low_4w_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine if price is near 4-week levels (within 1% tolerance)
        near_high = np.abs(high[i] - high_4w_aligned[i]) / high_4w_aligned[i] < 0.01
        near_low = np.abs(low[i] - low_4w_aligned[i]) / low_4w_aligned[i] < 0.01

        if position == 0:
            # LONG: near 4-week low + bullish rejection + weekly uptrend + volume
            if near_low and bullish_rejection[i] and weekly_uptrend_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: near 4-week high + bearish rejection + weekly downtrend + volume
            elif near_high and bearish_rejection[i] and weekly_downtrend_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches 4-week high OR weekly trend turns down
            if near_high or weekly_downtrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches 4-week low OR weekly trend turns up
            if near_low or weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals