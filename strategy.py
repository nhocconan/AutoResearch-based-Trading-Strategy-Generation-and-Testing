#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_Volume
# Hypothesis: Use 12h EMA34 for trend direction, Camarilla R1/S1 levels for breakout entries, and volume spike for confirmation.
# Long when price breaks above R1 with volume > 2x average in uptrend (price > 12h EMA34).
# Short when price breaks below S1 with volume > 2x average in downtrend (price < 12h EMA34).
# Exit when price returns to Camarilla Pivot level or trend reverses.
# Designed for low trade frequency (~30-50/year) to avoid fee drift and work in both bull and bear markets via trend filter.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_Volume"
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

    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)

    # Calculate Camarilla levels from previous day
    # Use daily high, low, close from 1d data
    df_1d = get_htf_data(prices, '1d')
    # Ensure we have enough data
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Align to 4h timeframe
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_4h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    # Pivot = (High + Low + Close) / 3
    rng = prev_high_4h - prev_low_4h
    r1 = prev_close_4h + rng * 1.1 / 12
    s1 = prev_close_4h - rng * 1.1 / 12
    pivot = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0

    # Volume filter: >2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(pivot[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 with volume spike in uptrend
            if (close[i] > r1[i] and close[i-1] <= r1[i-1] and  # Breakout condition
                volume[i] > vol_avg_20[i] * 2.0 and
                close[i] > ema34_12h_aligned[i]):  # Uptrend filter
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with volume spike in downtrend
            elif (close[i] < s1[i] and close[i-1] >= s1[i-1] and  # Breakdown condition
                  volume[i] > vol_avg_20[i] * 2.0 and
                  close[i] < ema34_12h_aligned[i]):  # Downtrend filter
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to pivot or trend reverses
            if close[i] <= pivot[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to pivot or trend reverses
            if close[i] >= pivot[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals