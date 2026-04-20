#!/usr/bin/env python3
"""
6h_LongTerm_Donchian_Breakout_With_WeeklyTrend_Filter
Hypothesis: Use 6h Donchian(20) breakouts with weekly trend filter (price > weekly EMA50 for longs, < weekly EMA50 for shorts) and volume confirmation.
Weekly trend filter ensures we only trade in the direction of the higher timeframe trend, reducing counter-trend whipsaws in both bull and bear markets.
Volume confirmation filters out low-momentum breakouts. Target: 12-37 trades/year with position size 0.25.
Works in bull/bear: Weekly trend filter adapts to long-term market direction, avoiding counter-trend trades during trend reversals.
"""

name = "6h_LongTerm_Donchian_Breakout_With_WeeklyTrend_Filter"
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
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = multiplier * close_weekly[i] + (1 - multiplier) * ema50_weekly[i-1]
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume filter (volume > 1.5x 30-period average)
    vol_ma30 = np.full_like(volume, np.nan)
    for i in range(30, n):
        vol_ma30[i] = np.mean(volume[i-30:i])
    volume_filter = volume > (1.5 * vol_ma30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_weekly_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter AND weekly uptrend
            if close[i] > donchian_high[i] and volume_filter[i] and close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume filter AND weekly downtrend
            elif close[i] < donchian_low[i] and volume_filter[i] and close[i] < ema50_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR weekly trend turns down
            if close[i] < donchian_low[i] or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR weekly trend turns up
            if close[i] > donchian_high[i] or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals