#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_1dTrend_Volume
Hypothesis: Trade weekly pivot breakouts on 6h timeframe when aligned with 1d EMA200 trend and confirmed by volume spike. Weekly pivots act as strong institutional support/resistance levels. In bull markets, buy breakouts above weekly R1; in bear markets, sell breakdowns below weekly S1. Volume confirmation filters false breakouts. Trend filter ensures we trade with the higher timeframe momentum. Designed for low trade frequency (15-30/year) to minimize fee drag while capturing significant moves.
Timeframe: 6h
"""

name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot levels ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    # Calculate weekly pivot points (using prior week's OHLC)
    ph_w = df_1w['high'].shift(1).values  # prior week high
    pl_w = df_1w['low'].shift(1).values   # prior week low
    pc_w = df_1w['close'].shift(1).values # prior week close
    pw = (ph_w + pl_w + pc_w) / 3.0       # weekly pivot
    r1_w = 2 * pw - pl_w                  # weekly resistance 1
    s1_w = 2 * pw - ph_w                  # weekly support 1

    # Align weekly pivots to 6h: constant through the week
    pw_aligned = align_htf_to_ltf(prices, df_1w, pw)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)

    # Get daily data for EMA200 trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # Volume spike: current > 2.5x average of last 4 bars (1 day on 6h)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after EMA200 warmup
        if (np.isnan(pw_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > weekly R1 + price > 1d EMA200 + volume spike
            if (close[i] > r1_w_aligned[i] and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: close < weekly S1 + price < 1d EMA200 + volume spike
            elif (close[i] < s1_w_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < weekly pivot P
            if close[i] < pw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close > weekly pivot P
            if close[i] > pw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals