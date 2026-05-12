#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels (R1, S1) from the prior day act as strong intraday support/resistance. 
Breakouts above R1 or below S1 with volume confirmation and weekly trend filter capture momentum moves.
Works in both bull and bear markets by aligning with the weekly trend direction.
"""

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "4h"
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

    # Get 1d data for Camarilla pivot calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values

    # Calculate Camarilla levels (R1, S1) from prior day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from the prior completed day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each day
    camarilla_width = (prev_high - prev_low) * 1.1 / 12
    r1 = prev_close + camarilla_width
    s1 = prev_close - camarilla_width
    
    # Align Camarilla levels to 4h timeframe (using prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 with volume and weekly uptrend
            if close[i] > r1_aligned[i] and volume[i] > vol_avg_20[i] * 1.5 and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume and weekly downtrend
            elif close[i] < s1_aligned[i] and volume[i] > vol_avg_20[i] * 1.5 and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns below R1 or weekly trend turns down
            if close[i] < r1_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above S1 or weekly trend turns up
            if close[i] > s1_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals