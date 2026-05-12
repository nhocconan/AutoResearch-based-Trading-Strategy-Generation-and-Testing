#!/usr/bin/env python3
"""
6h_Pivot_Squeeze_Breakout_1dTrend
Hypothesis: Price breaking above/below weekly pivot levels (calculated from weekly high-low-close) 
with 1d EMA trend filter and volatility contraction filter (BB width < 50th percentile) captures 
breakouts from consolidation with trend alignment. Works in bull/bear by following 1d trend direction.
Weekly pivots provide stronger support/resistance than daily, reducing false signals.
"""

name = "6h_Pivot_Squeeze_Breakout_1dTrend"
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

    # Calculate weekly pivot points (standard floor trader method)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values

    pivot_point = (weekly_high + weekly_low + weekly_close) / 3.0
    pivot_high = weekly_high
    pivot_low = weekly_low

    # Calculate support/resistance levels
    r1 = 2 * pivot_point - pivot_low
    s1 = 2 * pivot_point - pivot_high
    r2 = pivot_point + (pivot_high - pivot_low)
    s2 = pivot_point - (pivot_high - pivot_low)
    r3 = pivot_high + 2 * (pivot_point - pivot_low)
    s3 = pivot_low - 2 * (pivot_high - pivot_point)

    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)

    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Bollinger Band width contraction filter (20-period)
    bb_length = 20
    bb_mult = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    std_20 = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_bb = sma_20 + bb_mult * std_20
    lower_bb = sma_20 - bb_mult * std_20
    bb_width = upper_bb - lower_bb
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    vol_contract = bb_width < (0.5 * bb_width_ma)  # BB width < 50% of its MA

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after BB width MA warmup
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_contract[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly R3 + 1d EMA34 uptrend + volatility contraction
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_contract[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S3 + 1d EMA34 downtrend + volatility contraction
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_contract[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly S1 (reversal level)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly R1 (reversal level)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals