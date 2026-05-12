#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Breakouts at weekly Camarilla R1/S1 levels with volume confirmation and 1w trend filter.
Targets ~15-25 trades/year per symbol to minimize fee drag. Trend filter works in bull/bear markets.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
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

    # Get weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Calculate weekly Camarilla levels from previous weekly close
    camarilla_range = high_1w - low_1w
    r1 = close_1w + camarilla_range * 1.1 / 12
    s1 = close_1w - camarilla_range * 1.1 / 12

    # Get weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 12h bar (only uses completed weekly bars)
        r1_aligned = align_htf_to_ltf(prices, df_1w, r1)[i]
        s1_aligned = align_htf_to_ltf(prices, df_1w, s1)[i]
        ema50_aligned = ema50_1w_aligned[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(r1_aligned) or np.isnan(s1_aligned) or 
            np.isnan(ema50_aligned) or np.isnan(vol_threshold_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above weekly R1 + volume spike (1.5x) + 1w uptrend
            if (close[i] > r1_aligned and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below weekly S1 + volume spike (1.5x) + 1w downtrend
            elif (close[i] < s1_aligned and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly S1
            if close[i] < s1_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly R1
            if close[i] > r1_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals