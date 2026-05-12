#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1S1_Breakout_Volume_Switch
Hypothesis: On 1h, breakouts of 4h/1d Camarilla R1/S1 levels trigger entries when confirmed by multi-timeframe trend alignment (4h EMA > 1d EMA for long, < for short) and volume surge (1.5x 20-period average). Exits on opposite level breach or trend flip. Designed for low-frequency, high-conviction trades in both bull and bear markets via trend filter and volume confirmation. Targets 15-35 trades/year.
"""

name = "1h_4h1d_Camarilla_R1S1_Breakout_Volume_Switch"
timeframe = "1h"
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

    # Get 4h and 1d data (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)

    # Calculate 4h Camarilla R1/S1
    hh_4h = df_4h['high'].values
    ll_4h = df_4h['low'].values
    cc_4h = df_4h['close'].values
    r1_4h = cc_4h + (hh_4h - ll_4h) * 1.1 / 12
    s1_4h = cc_4h - (hh_4h - ll_4h) * 1.1 / 12

    # Calculate 1d Camarilla R1/S1
    hh_1d = df_1d['high'].values
    ll_1d = df_1d['low'].values
    cc_1d = df_1d['close'].values
    r1_1d = cc_1d + (hh_1d - ll_1d) * 1.1 / 12
    s1_1d = cc_1d - (hh_1d - ll_1d) * 1.1 / 12

    # Calculate 4h EMA20 for trend
    close_4h = pd.Series(cc_4h)
    ema20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate 1d EMA20 for trend
    close_1d = pd.Series(cc_1d)
    ema20_1d = close_1d.ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 1h bar
        r1_4h_a = align_htf_to_ltf(prices, df_4h, r1_4h)[i]
        s1_4h_a = align_htf_to_ltf(prices, df_4h, s1_4h)[i]
        r1_1d_a = align_htf_to_ltf(prices, df_1d, r1_1d)[i]
        s1_1d_a = align_htf_to_ltf(prices, df_1d, s1_1d)[i]
        ema20_4h_a = align_htf_to_ltf(prices, df_4h, ema20_4h)[i]
        ema20_1d_a = align_htf_to_ltf(prices, df_1d, ema20_1d)[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(r1_4h_a) or np.isnan(s1_4h_a) or np.isnan(r1_1d_a) or 
            np.isnan(s1_1d_a) or np.isnan(ema20_4h_a) or np.isnan(ema20_1d_a) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above 4h R1 with bullish 4h>1d trend and volume surge
            if (close[i] > r1_4h_a and 
                ema20_4h_a > ema20_1d_a and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below 4h S1 with bearish 4h<1d trend and volume surge
            elif (close[i] < s1_4h_a and 
                  ema20_4h_a < ema20_1d_a and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 4h S1 or trend flips bearish
            if (close[i] < s1_4h_a or ema20_4h_a < ema20_1d_a):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 4h R1 or trend flips bullish
            if (close[i] > r1_4h_a or ema20_4h_a > ema20_1d_a):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals