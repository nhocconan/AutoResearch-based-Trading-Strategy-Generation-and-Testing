#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_TrendFilter
Hypothesis: Use weekly pivot point direction (based on weekly close) to set bias, then trade Donchian(20) breakouts on 6h timeframe only in the direction of the weekly trend, with volume confirmation. This combines weekly structural bias with intermediate-term breakout logic, designed to work in both bull and bear markets by aligning with the higher timeframe trend. Targets 15-35 trades/year to avoid fee drag.
"""

name = "6h_WeeklyPivot_DonchianBreakout_TrendFilter"
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
    
    # Get weekly data for pivot calculation and trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader method)
    # Pivot = (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly bias: 1 if close > pivot (bullish), -1 if close < pivot (bearish)
    weekly_bias = np.where(weekly_close > weekly_pivot, 1, -1)
    # Align weekly bias to 6h timeframe (wait for weekly bar to close)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Calculate Donchian channels on 6h data (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Weekly bullish bias + price breaks above Donchian high + volume confirmation
            if weekly_bias_aligned[i] == 1 and high[i] > highest_high[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly bearish bias + price breaks below Donchian low + volume confirmation
            elif weekly_bias_aligned[i] == -1 and low[i] < lowest_low[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or weekly bias turns bearish
            if low[i] < lowest_low[i] or weekly_bias_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or weekly bias turns bullish
            if high[i] > highest_high[i] or weekly_bias_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals