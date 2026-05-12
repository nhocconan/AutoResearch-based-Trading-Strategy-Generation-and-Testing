#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_TrendFilter
Hypothesis: Elder Ray's Bull/Bear Power (EMA13-based) combined with 1d trend filter and volume confirmation 
captures institutional momentum while filtering false signals. Works in both bull and bear markets by 
aligning with higher timeframe direction and requiring volume validation.
Target: 15-35 trades/year (60-140 total over 4 years) with low turnover to minimize fee drag.
"""

name = "6h_ElderRay_BullBearPower_TrendFilter"
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

    # Get 1d data for trend filter and Elder Ray calculation (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate EMA13 for Elder Ray (13-period EMA of close)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power (High - EMA13) and Bear Power (Low - EMA13)
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 6h bar
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_avg_val = vol_avg_20[i]
        close_price = close[i]

        # Skip if any required data is NaN
        if (np.isnan(bull) or np.isnan(bear) or 
            np.isnan(ema50) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Strong Bull Power (>0) + price above EMA50 + volume surge
            if (bull > 0 and 
                close_price > ema50 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Strong Bear Power (<0) + price below EMA50 + volume surge
            elif (bear < 0 and 
                  close_price < ema50 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power turns negative or price below EMA50
            if (bear < 0 or close_price < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power turns positive or price above EMA50
            if (bull > 0 or close_price > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals