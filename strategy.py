#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Pullback_Trend_1dVolume
Hypothesis: In 6h timeframe, price often pulls back to weekly pivot levels (R1/S1) before continuing in the direction of the weekly trend. 
Long when price pulls back to weekly S1 in a weekly uptrend with volume confirmation (>1.2x 20-period avg). 
Short when price pulls back to weekly R1 in a weekly downtrend with volume confirmation.
Exit when price reaches the opposite weekly pivot level (R2/S2) or weekly trend reverses.
Designed for low trade frequency (<30/year) to minimize fee dust while capturing trend continuation moves.
Works in both bull and bear markets by using weekly trend filter.
"""

name = "6h_Weekly_Pivot_Pullback_Trend_1dVolume"
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

    # Get weekly data for pivot calculation and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values

    # Weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)

    # Weekly trend: price > weekly close = uptrend, price < weekly close = downtrend
    weekly_trend = np.where(close_1w > np.roll(close_1w, 1), 1, -1)  # 1=up, -1=down
    weekly_trend[0] = 1  # initialize

    # Align weekly data to 6h
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)

    # Volume confirmation: 1d volume > 1.2x 20-day average
    vol_avg_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or \
           np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or np.isnan(weekly_trend_aligned[i]) or \
           np.isnan(volume_1d_aligned[i]) or np.isnan(vol_avg_20d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Pullback to S1 in weekly uptrend with volume confirmation
            if (weekly_trend_aligned[i] == 1 and 
                low[i] <= s1_1w_aligned[i] * 1.005 and  # Allow small buffer for wicks
                volume_1d_aligned[i] > vol_avg_20d_aligned[i] * 1.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback to R1 in weekly downtrend with volume confirmation
            elif (weekly_trend_aligned[i] == -1 and 
                  high[i] >= r1_1w_aligned[i] * 0.995 and  # Allow small buffer for wicks
                  volume_1d_aligned[i] > vol_avg_20d_aligned[i] * 1.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches S2 or weekly trend turns down
            if low[i] <= s2_1w_aligned[i] or weekly_trend_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches R2 or weekly trend turns up
            if high[i] >= r2_1w_aligned[i] or weekly_trend_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals