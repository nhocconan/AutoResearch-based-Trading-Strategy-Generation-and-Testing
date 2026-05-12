#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Trend
Hypothesis: Uses weekly Camarilla pivot levels (R1/S1) for breakout entries on the daily timeframe,
confirmed by weekly EMA20 trend and volume surge. Designed to capture multi-week moves with low trade frequency.
Works in both bull and bear markets by adapting to weekly trend context.
"""

name = "1d_Weekly_Pivot_R1_S1_Breakout_Trend"
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

    # Get weekly data for Camarilla pivot points and EMA20 (call once before loop)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)

    # Calculate weekly Camarilla pivot levels (standard formula)
    hh_w = df_w['high'].values
    ll_w = df_w['low'].values
    cc_w = df_w['close'].values

    # Camarilla levels
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    r1_w = cc_w + (hh_w - ll_w) * 1.1 / 12
    s1_w = cc_w - (hh_w - ll_w) * 1.1 / 12

    # Calculate weekly EMA20 for trend filter
    close_w = pd.Series(cc_w)
    ema20_w = close_w.ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume confirmation: 5-period average (1 week of daily data)
    vol_avg_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(5, n):  # Start from 5 to have enough data for volume average
        # Get aligned values for current daily bar
        r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)[i]
        s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)[i]
        ema20_w_aligned = align_htf_to_ltf(prices, df_w, ema20_w)[i]
        vol_avg_val = vol_avg_5[i]
        
        # Skip if any required data is NaN
        if (np.isnan(r1_w_aligned) or np.isnan(s1_w_aligned) or 
            np.isnan(ema20_w_aligned) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above weekly R1 with bullish weekly trend and volume surge
            if (close[i] > r1_w_aligned and 
                close[i] > ema20_w_aligned and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S1 with bearish weekly trend and volume surge
            elif (close[i] < s1_w_aligned and 
                  close[i] < ema20_w_aligned and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below weekly S1 or EMA20 (reversal signal)
            if (close[i] < s1_w_aligned or close[i] < ema20_w_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above weekly R1 or EMA20 (reversal signal)
            if (close[i] > r1_w_aligned or close[i] > ema20_w_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals