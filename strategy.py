#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) for breakout entries on the 12-hour timeframe,
confirmed by daily EMA34 trend and volume surge. Designed to capture multi-day moves with low trade frequency.
Works in both bull and bear markets by adapting to daily trend context.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend"
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

    # Get daily data for Camarilla pivot points and EMA34 (call once before loop)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 34:
        return np.zeros(n)

    # Calculate daily Camarilla pivot levels (standard formula)
    hh_d = df_d['high'].values
    ll_d = df_d['low'].values
    cc_d = df_d['close'].values

    # Camarilla levels
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    r1_d = cc_d + (hh_d - ll_d) * 1.1 / 12
    s1_d = cc_d - (hh_d - ll_d) * 1.1 / 12

    # Calculate daily EMA34 for trend filter
    close_d = pd.Series(cc_d)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values

    # Volume confirmation: 5-period average (5 days of daily data)
    vol_avg_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(5, n):  # Start from 5 to have enough data for volume average
        # Get aligned values for current 12h bar
        r1_d_aligned = align_htf_to_ltf(prices, df_d, r1_d)[i]
        s1_d_aligned = align_htf_to_ltf(prices, df_d, s1_d)[i]
        ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)[i]
        vol_avg_val = vol_avg_5[i]
        
        # Skip if any required data is NaN
        if (np.isnan(r1_d_aligned) or np.isnan(s1_d_aligned) or 
            np.isnan(ema34_d_aligned) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above daily R1 with bullish daily trend and volume surge
            if (close[i] > r1_d_aligned and 
                close[i] > ema34_d_aligned and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below daily S1 with bearish daily trend and volume surge
            elif (close[i] < s1_d_aligned and 
                  close[i] < ema34_d_aligned and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below daily S1 or EMA34 (reversal signal)
            if (close[i] < s1_d_aligned or close[i] < ema34_d_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above daily R1 or EMA34 (reversal signal)
            if (close[i] > r1_d_aligned or close[i] > ema34_d_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals