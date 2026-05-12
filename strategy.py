#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend
Hypothesis: On 6h timeframe, buy when price breaks above Camarilla R4 with price > 1d EMA34 and volume > 1.5x average; sell when price breaks below Camarilla S4 with price < 1d EMA34 and volume > 1.5x average. Uses Camarilla levels from daily timeframe for strong support/resistance, trend filter from 1d EMA, and volume confirmation to avoid false breakouts. Targets strong trending moves in both bull and bear markets with low trade frequency to minimize fee drag.
"""

name = "6h_Camarilla_R4_S4_Breakout_1dTrend"
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

    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R4 = prev_close + range_hl * 1.1 / 2
    S4 = prev_close - range_hl * 1.1 / 2
    R3 = prev_close + range_hl * 1.1 / 4
    S3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R4 + 1d uptrend + volume spike
            if (close[i] > R4_aligned[i-1] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 + 1d downtrend + volume spike
            elif (close[i] < S4_aligned[i-1] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR trend turns down
            if close[i] < S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR trend turns up
            if close[i] > R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals