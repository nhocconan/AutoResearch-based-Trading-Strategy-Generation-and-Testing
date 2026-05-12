#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R1/S1) from 1d timeframe provide institutional-grade support/resistance. 
Breakout above R1 or below S1 with 1d EMA34 trend filter and volume spike (>1.5x 20-period avg) captures strong directional moves. 
Exit on touch of opposite Camarilla level (S1 for longs, R1 for shorts) or trend reversal.
Designed for low trade frequency (20-40/year) to minimize fee decay while capturing institutional flow.
Works in both bull/bear markets: breaks signal institutional participation regardless of trend direction.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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

    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels (R1, S1) from previous 1d bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We use previous day's values to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First day has no previous, so set to NaN
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    rang = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * rang / 12
    s1 = prev_close_1d - 1.1 * rang / 12

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Check if we have valid data
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 1d uptrend + volume spike
            if close[i] > r1[i-1] and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 1d downtrend + volume spike
            elif close[i] < s1[i-1] and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches or goes below S1 OR trend turns down
            if close[i] <= s1[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches or goes above R1 OR trend turns up
            if close[i] >= r1[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals