#!/usr/bin/env python3
"""
6h_LongTermTrend_Fade_Retest
Hypothesis: In multi-year trends (bull/bear), price respects long-term trendlines from weekly highs/lows.
Enter on retest of weekly trendline in direction of 1d trend, with volume confirmation.
Exit on trendline break or trend reversal. Works in both bull and bear by following weekly structure.
"""

name = "6h_LongTermTrend_Fade_Retest"
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
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly higher highs and lower lows for trendline
    # Use 2-period lookback for swing points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Weekly swing highs (local maxima)
    swing_high = np.zeros_like(high_1w, dtype=bool)
    swing_low = np.zeros_like(low_1w, dtype=bool)
    for i in range(2, len(high_1w)-2):
        if high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and \
           high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]:
            swing_high[i] = True
        if low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and \
           low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]:
            swing_low[i] = True

    # Get most recent swing points
    last_swing_high_idx = np.where(swing_high)[0]
    last_swing_low_idx = np.where(swing_low)[0]
    
    # Initialize arrays for trendlines
    weekly_resistance = np.full(len(close_1w), np.nan)
    weekly_support = np.full(len(close_1w), np.nan)
    
    # For each point, calculate trendline from last swing
    for i in range(len(close_1w)):
        # Resistance trendline: connect last two swing highs
        if len(last_swing_high_idx[last_swing_high_idx <= i]) >= 2:
            idx1, idx2 = last_swing_high_idx[last_swing_high_idx <= i][-2:]
            if idx1 != idx2:  # Avoid division by zero
                # Linear interpolation: y = m*x + b
                x1, y1 = idx1, high_1w[idx1]
                x2, y2 = idx2, high_1w[idx2]
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                weekly_resistance[i] = m * i + b
        
        # Support trendline: connect last two swing lows
        if len(last_swing_low_idx[last_swing_low_idx <= i]) >= 2:
            idx1, idx2 = last_swing_low_idx[last_swing_low_idx <= i][-2:]
            if idx1 != idx2:
                x1, y1 = idx1, low_1w[idx1]
                x2, y2 = idx2, low_1w[idx2]
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                weekly_support[i] = m * i + b

    # Align trendlines to 6h timeframe
    weekly_resistance_aligned = align_htf_to_ltf(prices, df_1w, weekly_resistance)
    weekly_support_aligned = align_htf_to_ltf(prices, df_1w, weekly_support)

    # 1d EMA50 trend filter (for entry direction)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume confirmation: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(weekly_resistance_aligned[i]) or np.isnan(weekly_support_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price retests weekly support trendline in uptrend
            if (close[i] > weekly_support_aligned[i] * 0.998 and  # Allow 0.2% tolerance
                close[i] < weekly_support_aligned[i] * 1.002 and
                close[i] > ema_50_1d_aligned[i] and  # In uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price retests weekly resistance trendline in downtrend
            elif (close[i] < weekly_resistance_aligned[i] * 1.002 and  # Allow 0.2% tolerance
                  close[i] > weekly_resistance_aligned[i] * 0.998 and
                  close[i] < ema_50_1d_aligned[i] and  # In downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly support OR trend turns down
            if close[i] < weekly_support_aligned[i] * 0.995 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly resistance OR trend turns up
            if close[i] > weekly_resistance_aligned[i] * 1.005 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals