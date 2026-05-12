#!/usr/bin/env python3
"""
6h_WeeklyPivot_Reversal
Hypothesis: In range-bound or weak-trend markets (common in 2025+), price often reverses at weekly pivot levels (R1/S1, R2/S2). This strategy fades extreme weekly pivot touches (R3/S3) with 12h trend filter and volume exhaustion signals. Works in both bull/bear by fading overextended moves regardless of direction.
"""

name = "6h_WeeklyPivot_Reversal"
timeframe = "6h"
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

    # Get weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')

    # Calculate weekly high, low, close for pivot levels
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values

    # Weekly pivot point and support/resistance levels
    # Pivot = (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    # R2 = P + (H - L), S2 = P - (H - L)
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)

    # Align weekly levels to 6h timeframe
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)

    # 12h trend filter (EMA34) - only trade against the trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)

    # Volume exhaustion: current volume < 50% of 20-period average (sign of selling/buying climax exhaustion)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_exhaustion = volume < (0.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(35, n):  # Start after EMA warmup
        if (np.isnan(r3_w_aligned[i]) or np.isnan(s3_w_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_exhaustion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches or goes below S3 (extreme oversold) + 12h trend is down (fade the move) + volume exhaustion
            if (low[i] <= s3_w_aligned[i] and 
                close[i] < ema_34_12h_aligned[i] and 
                volume_exhaustion[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above R3 (extreme overbought) + 12h trend is up (fade the move) + volume exhaustion
            elif (high[i] >= r3_w_aligned[i] and 
                  close[i] > ema_34_12h_aligned[i] and 
                  volume_exhaustion[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot (mean reversion complete) or stop if trend resumes
            if close[i] >= pivot_w_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot or stop if trend resumes
            if close[i] <= pivot_w_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals