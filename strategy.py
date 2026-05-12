#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h breakouts of daily Camarilla R1/S1 levels trigger entries when confirmed by 1d EMA trend alignment and volume surge (>1.5x 20-period average). Exits on opposite level breach or trend flip. Designed for low-frequency, high-conviction trades in both bull and bear markets via trend filter and volume confirmation. Targets 20-40 trades/year.
"""

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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

    # Get daily data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate daily Camarilla R1/S1
    hh_1d = df_1d['high'].values
    ll_1d = df_1d['low'].values
    cc_1d = df_1d['close'].values
    r1_1d = cc_1d + (hh_1d - ll_1d) * 1.1 / 12
    s1_1d = cc_1d - (hh_1d - ll_1d) * 1.1 / 12

    # Calculate daily EMA20 for trend
    close_1d = pd.Series(cc_1d)
    ema20_1d = close_1d.ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 4h bar
        r1_1d_a = align_htf_to_ltf(prices, df_1d, r1_1d)[i]
        s1_1d_a = align_htf_to_ltf(prices, df_1d, s1_1d)[i]
        ema20_1d_a = align_htf_to_ltf(prices, df_1d, ema20_1d)[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(r1_1d_a) or np.isnan(s1_1d_a) or np.isnan(ema20_1d_a) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above daily R1 with bullish trend and volume surge
            if (close[i] > r1_1d_a and 
                ema20_1d_a > close[i] and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below daily S1 with bearish trend and volume surge
            elif (close[i] < s1_1d_a and 
                  ema20_1d_a < close[i] and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below daily S1 or trend turns bearish
            if (close[i] < s1_1d_a or ema20_1d_a < close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above daily R1 or trend turns bullish
            if (close[i] > r1_1d_a or ema20_1d_a > close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals