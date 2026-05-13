#!/usr/bin/env python3
# 6h_Weekly_Pivot_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Use weekly pivot points R3/S3 (stronger support/resistance than R1/S1) for breakout entries.
# Enter long when price breaks above weekly R3 in weekly uptrend with volume spike.
# Enter short when price breaks below weekly S3 in weekly downtrend with volume spike.
# Exit when price returns to weekly pivot point (PP).
# Weekly pivots reduce false breakouts; R3/S3 levels require stronger momentum, improving win rate in both bull and bear markets.

name = "6h_Weekly_Pivot_R3_S3_Breakout_1wTrend_Volume"
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

    # Get weekly data for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP, R1, S1, R2, S2, R3, S3
    # Standard pivot point formulas:
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    # R3 = H + 2*(PP - L)
    # S3 = L - 2*(H - PP)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pp_1w = (high_w + low_w + close_w) / 3
    r1_1w = (2 * pp_1w) - low_w
    s1_1w = (2 * pp_1w) - high_w
    r2_1w = pp_1w + (high_w - low_w)
    s2_1w = pp_1w - (high_w - low_w)
    r3_1w = high_w + 2 * (pp_1w - low_w)
    s3_1w = low_w - 2 * (high_w - pp_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)

    # Weekly trend: price above/below weekly EMA200
    ema_200_1w = pd.Series(close_w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + price above weekly EMA200 (uptrend) + volume spike
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema_200_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price below weekly EMA200 (downtrend) + volume spike
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP)
            if close[i] <= pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP)
            if close[i] >= pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals