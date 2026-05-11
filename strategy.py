#!/usr/bin/env python3
"""
6h_Angle_of_Descent_v1
Hypothesis: Measures the angle of price descent from the 1d high using a 12h window.
In bull markets, steep declines often reverse sharply; in bear markets, shallow declines
continue the trend. Uses 1d high as reference and 12h EMA for trend filter.
Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
"""

name = "6h_Angle_of_Descent_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D Data for Reference High ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    # Previous day's high (reference point)
    ref_high_1d = high_1d  # this is the prior day's high
    
    # Align reference high to 6h
    ref_high_aligned = align_htf_to_ltf(prices, df_1d, ref_high_1d)
    
    # === 12H Data for Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA21 for trend
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # === Calculate Angle of Descent ===
    # Angle = arctan((current price - reference high) / time_in_bars)
    # We use 12 bars (3 days) lookback for the angle calculation
    lookback = 12  # 12 * 6h = 3 days
    angle_of_descent = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if np.isnan(ref_high_aligned[i]):
            continue
        price_change = close[i] - ref_high_aligned[i]
        # Normalize by reference price to get percentage change
        price_change_pct = price_change / ref_high_aligned[i]
        # Angle in degrees: arctan(price_change_pct * 100) * (180/pi) 
        # Multiply by 100 to get reasonable angle values
        angle = np.arctan(price_change_pct * 100) * (180 / np.pi)
        angle_of_descent[i] = angle
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(100, lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ref_high_aligned[i]) or 
            np.isnan(ema21_12h_aligned[i]) or 
            np.isnan(angle_of_descent[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: shallow angle of descent (> -10 degrees) in downtrend (mean reversion bounce)
            # OR steep angle of descent (< -30 degrees) in uptrend (panic sell exhaustion)
            if (angle_of_descent[i] > -10 and ema21_12h_aligned[i] < close[i]) or \
               (angle_of_descent[i] < -30 and ema21_12h_aligned[i] > close[i]):
                signals[i] = 0.25
                position = 1
            # Short: steep angle of descent (< -30 degrees) in downtrend (continuation)
            elif angle_of_descent[i] < -30 and ema21_12h_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: angle becomes too steep (> -30) indicating continued weakness
            if angle_of_descent[i] < -30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: angle flattens (> -10) indicating loss of momentum
            if angle_of_descent[i] > -10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals