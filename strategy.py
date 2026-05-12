#!/usr/bin/env python3
# 1d_WilliamsAlligator_Trend_Filter
# Hypothesis: Williams Alligator on 1d for trend direction with 1h Williams %R for entry timing.
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend and avoid chop.
# Entry when price aligns with Alligator direction and Williams %R shows momentum.
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull/bear by following 1d trend with Alligator filter.

name = "1d_WilliamsAlligator_Trend_Filter"
timeframe = "1d"
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

    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Williams Alligator: SMAs of median price
    median_price_1d = (high_1d + low_1d) / 2
    # Jaw: 13-period SMMA shifted 8 bars
    jaw_raw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    # Teeth: 8-period SMMA shifted 5 bars
    teeth_raw = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    # Lips: 5-period SMMA shifted 3 bars
    lips_raw = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan

    # Align Alligator lines to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)

    # Get 1h data for Williams %R entry timing
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)

    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values

    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1h = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    lowest_low_1h = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_1h - close_1h) / (highest_high_1h - lowest_low_1h)
    # Handle division by zero
    williams_r = np.where((highest_high_1h - lowest_low_1h) == 0, -50, williams_r)

    # Align Williams %R to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1h, williams_r)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) and Williams %R oversold (< -80)
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and williams_r_aligned[i] < -80:
                signals[i] = 0.25
                position = 1
            # SHORT: Jaws > Teeth > Lips (bearish alignment) and Williams %R overbought (> -20)
            elif jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and williams_r_aligned[i] > -20:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks or Williams %R overbought
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks or Williams %R oversold
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals