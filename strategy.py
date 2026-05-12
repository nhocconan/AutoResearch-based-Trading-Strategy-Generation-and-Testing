#!/usr/bin/env python3
"""
6h_WeeklyPivot_Zone_Reversal
Hypothesis: In 6-hour timeframe, price reverses from weekly pivot support/resistance zones (S1/R1, S2/R2) with volume confirmation.
Works in bull/bear by fading extreme weekly levels, avoiding whipsaws via volume filter.
Targets 15-30 trades/year by requiring confluence of price at pivot zone + volume spike.
"""

name = "6h_WeeklyPivot_Zone_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivots(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L, etc."""
    P = (high + low + close) / 3.0
    S1 = 2 * P - high
    R1 = 2 * P - low
    S2 = P - (high - low)
    R2 = P + (high - low)
    return P, S1, R1, S2, R2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    # Calculate weekly pivot points
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values

    P, S1, R1, S2, R2 = calculate_weekly_pivots(wk_high, wk_low, wk_close)

    # Align pivot levels to 6h timeframe
    P_aligned = align_htf_to_ltf(prices, df_1w, P)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)

    # Volume spike: current > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after volume MA warmup
        if (np.isnan(P_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or
            np.isnan(S2_aligned[i]) or np.isnan(R2_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        price = close[i]

        if position == 0:
            # LONG: price near S1 or S2 (within 0.5%) + volume spike
            near_S1 = abs(price - S1_aligned[i]) / S1_aligned[i] < 0.005
            near_S2 = abs(price - S2_aligned[i]) / S2_aligned[i] < 0.005
            if (near_S1 or near_S2) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price near R1 or R2 (within 0.5%) + volume spike
            elif (abs(price - R1_aligned[i]) / R1_aligned[i] < 0.005 or
                  abs(price - R2_aligned[i]) / R2_aligned[i] < 0.005) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses above pivot point (P)
            if price > P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses below pivot point (P)
            if price < P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals