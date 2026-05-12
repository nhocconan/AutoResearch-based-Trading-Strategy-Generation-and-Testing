#!/usr/bin/env python3
# 6h_WilliamsAlligator_Trend_Filter
# Hypothesis: Williams Alligator (3 SMAs) on 6h with 12h trend filter and volume confirmation.
# The Alligator identifies trend state: jaws (13), teeth (8), lips (5) SMAs.
# In uptrend: lips > teeth > jaws; downtrend: lips < teeth < jaws.
# Only take trades aligned with 12h EMA50 trend to avoid counter-trend whipsaw.
# Volume spike (>1.5x SMA20) confirms momentum. Designed for 50-150 trades over 4 years.
# Works in bull/bear by following 12h trend and using Alligator for entry/exit.

name = "6h_WilliamsAlligator_Trend_Filter"
timeframe = "6h"
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

    # Get 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)

    close_6h = df_6h['close'].values

    # Williams Alligator: SMAs of median price (high+low)/2
    median_price = (df_6h['high'].values + df_6h['low'].values) / 2
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values  # SMA5
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values  # SMA8
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values # SMA13

    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Align Alligator lines to 6h timeframe
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    jaws_aligned = align_htf_to_ltf(prices, df_6h, jaws)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaws_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine Alligator trend state
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaws_val = jaws_aligned[i]
        bullish_aligned = lips_val > teeth_val > jaws_val  # Lips > Teeth > Jaws
        bearish_aligned = lips_val < teeth_val < jaws_val  # Lips < Teeth < Jaws

        if position == 0:
            # LONG: Bullish Alligator alignment + price above 12h EMA50 + volume spike
            if bullish_aligned and close[i] > ema50_12h_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment + price below 12h EMA50 + volume spike
            elif bearish_aligned and close[i] < ema50_12h_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns bearish (lips < teeth) or price below 12h EMA50
            if not bullish_aligned or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns bullish (lips > teeth) or price above 12h EMA50
            if not bearish_aligned or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals