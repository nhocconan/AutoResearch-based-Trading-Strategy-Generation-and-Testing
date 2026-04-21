#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_VolumeFilter
Hypothesis: Combines weekly pivot levels (trend filter from 1w) with Donchian(20) breakout on 6h for entry timing. Uses volume confirmation (>1.3x 20-bar MA) to reduce false breakouts. Weekly pivot provides structural bias that works in both bull (breakouts above R1) and bear (breakdowns below S1) markets. Target: 12-30 trades/year per symbol (50-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_open = np.roll(open_1w, 1)
    
    # First bar: use same values (will be filtered by min_periods later)
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    prev_open[0] = open_1w[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    weekly_r1 = 2 * pivot - prev_low
    weekly_s1 = 2 * pivot - prev_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian(20) channels on 6h timeframe
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average on 6h timeframe
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.3x average to reduce trades)
        volume_ok = volume > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian(20) high AND above weekly R1 (bullish bias)
            if price > donchian_high[i] and price > weekly_r1_aligned[i]:
                if volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Donchian(20) low AND below weekly S1 (bearish bias)
            elif price < donchian_low[i] and price < weekly_s1_aligned[i]:
                if volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price closes below Donchian(20) low (trend reversal) or below weekly S1
            if price < donchian_low[i] or price < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Donchian(20) high (trend reversal) or above weekly R1
            if price > donchian_high[i] or price > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_VolumeFilter"
timeframe = "6h"
leverage = 1.0