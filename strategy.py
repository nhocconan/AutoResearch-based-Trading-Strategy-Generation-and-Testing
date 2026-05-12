#!/usr/bin/env python3
"""
12h_Pivot_Reversal_VolumeSurge
Hypothesis: In both bull and bear markets, price often reverses near daily pivot points (PP, S1, R1) when accompanied by a volume surge. 
We calculate daily Camarilla pivot levels (based on prior day's OHLC), wait for price to touch these levels with volume > 2x 20-period average, 
and enter in the direction of the bounce. Uses 1w EMA50 trend filter to avoid counter-trend trades. Designed for low frequency (12h) to minimize fee drag.
"""

name = "12h_Pivot_Reversal_VolumeSurge"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: corrected function name

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter (more robust than 1d alone)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    # Calculate Camarilla pivot levels from prior day's OHLC
    # Typical price = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    daily_range = df_1d['high'] - df_1d['low']
    r1 = typical_price + daily_range * 1.1 / 12
    s1 = typical_price - daily_range * 1.1 / 12
    pp = typical_price  # Pivot Point

    # Align pivot levels to 12h timeframe (using prior day's values - no look-ahead)
    r1_aligned = align_ltf_to_htf(prices, df_1d, r1.values)
    s1_aligned = align_ltf_to_htf(prices, df_1d, s1.values)
    pp_aligned = align_ltf_to_htf(prices, df_1d, pp.values)

    # 1w EMA50 for trend filter (avoid counter-trend trades)
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_ltf_to_htf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup for volume average
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches S1 support with volume surge AND price > 1w EMA50 (uptrend filter)
            if low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and volume[i] > vol_avg_20[i] * 2.0 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R1 resistance with volume surge AND price < 1w EMA50 (downtrend filter)
            elif high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and volume[i] > vol_avg_20[i] * 2.0 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot point or trend breaks
            if high[i] >= pp_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot point or trend breaks
            if low[i] <= pp_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals