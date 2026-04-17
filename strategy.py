# SPDX-FileCopyrightText: 2025 Alpaca Trading Team
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily pivot levels to 12h timeframe (use previous day's levels)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume filter: current volume > 1.8 * 20-period average (tighter)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index filter (trending market filter)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((highest_high - lowest_low) / (atr_safe * 14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for volume MA and chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(volume_ma20[i]) or 
            np.isnan(chop[i]) or np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter (tighter)
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Choppiness filter: only trade in trending markets (CHOP < 38.2)
        trend_filter = chop[i] < 38.2
        
        # Weekly trend filter: price above/below weekly EMA34
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume, trend, and weekly uptrend
            if close[i] > r1_12h[i] and volume_filter and trend_filter and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume, trend, and weekly downtrend
            elif close[i] < s1_12h[i] and volume_filter and trend_filter and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1 (stop) OR chop increases (range developing) OR weekly trend turns down
            if close[i] < s1_12h[i] or chop[i] > 61.8 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1 OR chop increases (range developing) OR weekly trend turns up
            if close[i] > r1_12h[i] or chop[i] > 61.8 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA34_DailyPivot_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0