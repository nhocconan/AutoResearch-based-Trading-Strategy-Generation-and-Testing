#!/usr/bin/env python3
# 1d_WilliamsAlligator_Trend_Confirmation
# Hypothesis: Williams Alligator (three SMAs) identifies trend direction; trades occur when price aligns with Alligator jaws on daily timeframe with weekly trend filter and volume confirmation. Works in bull/bear by only taking trades in direction of weekly trend, reducing whipsaw. Targets 15-25 trades/year on daily timeframe to minimize fee drag.

name = "1d_WilliamsAlligator_Trend_Confirmation"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Williams Alligator on daily: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values

    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: 1.5x 20-day SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # start after jaw period
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Alligator alignment: Lips > Teeth > Jaw = uptrend; Lips < Teeth < Jaw = downtrend
        lips_gt_teeth = lips[i] > teeth[i]
        teeth_gt_jaw = teeth[i] > jaw[i]
        lips_lt_teeth = lips[i] < teeth[i]
        teeth_lt_jaw = teeth[i] < jaw[i]

        uptrend_aligned = lips_gt_teeth and teeth_gt_jaw
        downtrend_aligned = lips_lt_teeth and teeth_lt_jaw

        if position == 0:
            # LONG: Price > Lips, Alligator aligned up, weekly uptrend, volume spike
            if (close[i] > lips[i] and
                uptrend_aligned and
                close[i] > ema34_1w_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Lips, Alligator aligned down, weekly downtrend, volume spike
            elif (close[i] < lips[i] and
                  downtrend_aligned and
                  close[i] < ema34_1w_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Lips or weekly trend turns down
            if (close[i] < lips[i] or
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Lips or weekly trend turns up
            if (close[i] > lips[i] or
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals