#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly Camarilla pivot direction filter and volume confirmation.
- Donchian breakout: price > highest high of last 20 bars (long) or < lowest low of last 20 bars (short)
- Weekly Camarilla pivot: calculates R4/S4 levels from prior weekly OHLC; trend = long if close > weekly pivot, short if close < weekly pivot
- Volume confirmation: current volume > 1.5x 20-period average
- Entry: Donchian breakout in direction of weekly Camarilla trend + volume confirmation
- Exit: Opposite Donchian breakout or weekly trend flip
- Uses structure (Donchian) + HTF bias (weekly pivot) + volume for conviction
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly Camarilla pivot levels (R4, S4, pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior weekly OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Camarilla R4 = weekly_close + 1.5 * (weekly_high - weekly_low)
    camarilla_r4 = weekly_close + 1.5 * (weekly_high - weekly_low)
    # Camarilla S4 = weekly_close - 1.5 * (weekly_high - weekly_low)
    camarilla_s4 = weekly_close - 1.5 * (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Weekly trend: long if close > pivot, short if close < pivot
    weekly_trend = np.where(close > pivot_aligned, 1, np.where(close < pivot_aligned, -1, 0))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout signals
        donchian_long = close[i] > highest_high[i-1]  # Break above prior period high
        donchian_short = close[i] < lowest_low[i-1]   # Break below prior period low
        
        if position == 0:
            # Long entry: Donchian breakout up + weekly trend long + volume confirmation
            if donchian_long and weekly_trend[i] == 1 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakout down + weekly trend short + volume confirmation
            elif donchian_short and weekly_trend[i] == -1 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakout down OR weekly trend flip to short
            if donchian_short or weekly_trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up OR weekly trend flip to long
            if donchian_long or weekly_trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyCamarilla_PivotTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0