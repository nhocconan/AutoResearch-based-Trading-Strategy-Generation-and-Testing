#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter
Hypothesis: On 12h timeframe, price breaking above/below Camarilla R1/S1 levels indicates momentum continuation. Use 1w EMA50 as trend filter (price > EMA50 for long, < EMA50 for short) and volume > 2x 12-period average for confirmation. Exit on opposite Camarilla break or trend reversal. Targets 15-30 trades/year to avoid fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
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

    # Get 1w data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values

    # Calculate 1w EMA50 for trend
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Calculate 12h Camarilla levels (based on previous day's range)
    # For 12h chart, we use previous 24h (2 bars) high/low/close
    prev_high = np.roll(high, 2)  # Previous 12h bar's high
    prev_low = np.roll(low, 2)    # Previous 12h bar's low
    prev_close = np.roll(close, 2) # Previous 12h bar's close
    
    # Handle initial NaN values from roll
    prev_high[:2] = high[:2]
    prev_low[:2] = low[:2]
    prev_close[:2] = close[:2]
    
    # Camarilla levels calculation
    range_val = prev_high - prev_low
    R1 = prev_close + (range_val * 1.1 / 12)
    S1 = prev_close - (range_val * 1.1 / 12)

    # Volume confirmation: 2x 12-period average
    vol_avg_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period

    for i in range(50, n):
        # Get aligned values for current 12h bar
        ema50_a = align_htf_to_ltf(prices, df_1w, ema50)[i]
        vol_avg_val = vol_avg_12[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50_a) or np.isnan(vol_avg_val) or 
            np.isnan(R1[i]) or np.isnan(S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 + price above EMA50 + volume surge
            if (close[i] > R1[i] and 
                close[i] > ema50_a and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Price breaks below Camarilla S1 + price below EMA50 + volume surge
            elif (close[i] < S1[i] and 
                  close[i] < ema50_a and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            bars_since_entry += 1
            # EXIT LONG: Price breaks below Camarilla S1 or price below EMA50
            if (close[i] < S1[i] or close[i] < ema50_a):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            bars_since_entry += 1
            # EXIT SHORT: Price breaks above Camarilla R1 or price above EMA50
            if (close[i] > R1[i] or close[i] > ema50_a):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25

    return signals