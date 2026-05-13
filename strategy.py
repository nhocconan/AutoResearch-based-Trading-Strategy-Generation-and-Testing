#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_Trend_Filter
Hypothesis: Weekly pivot points (calculated from prior week) provide significant support/resistance.
Breakouts above R1 or below S1 with trend alignment (price vs weekly EMA) and volume confirmation
capture strong moves in both bull and bear markets. Weekly context filters noise, reducing false
breakouts. Targets 20-40 trades/year with discrete sizing to minimize fee drag.
"""

name = "6h_Weekly_Pivot_Breakout_Trend_Filter"
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
    
    # Calculate pivot points from weekly data (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use prior week's data to calculate pivots for current week
    # Shift by 1 to avoid look-ahead (use completed week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Calculate pivot points
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Trend filter: price vs weekly EMA (using prior week's EMA to avoid look-ahead)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().shift(1).values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R1 with bullish trend and volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema_20_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with bearish trend and volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema_20_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (reversal) or above R2 (take profit at stronger level)
            if close[i] < s1_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (reversal) or below S2 (take profit at stronger level)
            if close[i] > r1_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals