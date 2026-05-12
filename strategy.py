#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Trend_Confirmation_v1
Hypothesis: Williams Alligator (SMAs of median price) on daily chart defines trend; enter long when price > Alligator Teeth and Lips > Jaw, short when opposite, on 1d timeframe. Uses weekly trend filter to avoid counter-trend trades. Targets 15-25 trades/year to minimize fee drag. Works in bull (trend-following) and bear (avoids false signals via weekly filter).
"""

name = "1d_WilliamsAlligator_Trend_Confirmation_v1"
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

    # Williams Alligator on daily: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().values

    # Weekly trend filter: price > weekly EMA50 for uptrend, < for downtrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Align all to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # warmup for longest SMA
        # Skip if any data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(weekly_ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        weekly_trend = weekly_ema50_aligned[i]
        price = close[i]

        if position == 0:
            # LONG: Price above Teeth, Lips > Jaw (bullish alignment), and weekly uptrend
            if (price > teeth_val and lips_val > jaw_val and price > weekly_trend):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below Teeth, Lips < Jaw (bearish alignment), and weekly downtrend
            elif (price < teeth_val and lips_val < jaw_val and price < weekly_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Teeth or weekly trend turns down
            if price < teeth_val or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Teeth or weekly trend turns up
            if price > teeth_val or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals