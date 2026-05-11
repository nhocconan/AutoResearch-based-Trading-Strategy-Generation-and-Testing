#!/usr/bin/env python3
"""
6h_12h_Donchian_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: Uses 12h Donchian breakout with weekly pivot direction for trend filter and volume confirmation.
Works in both bull and bear markets by following higher-timeframe structure (weekly pivot) while using
12h Donchian for breakout signals. Targets low trade frequency (15-30/year) via strict confluence.
"""

name = "6h_12h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_weekly_pivot(high, low, close):
    """Calculate Weekly Pivot Points"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h Donchian Channel ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    donchian_upper_12h, donchian_lower_12h = calculate_donchian(
        df_12h['high'].values, df_12h['low'].values, period=20
    )
    
    # Align 12h Donchian to 6h timeframe
    donchian_upper_12h_6h = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_12h_6h = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    
    # --- Weekly Pivot Direction ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    pivot, r1, s1, r2, s2 = calculate_weekly_pivot(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    
    # Align weekly pivot to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly trend: 1 if price > pivot, -1 if price < pivot
    weekly_trend = np.where(close > pivot_6h, 1, -1)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h_6h[i]) or np.isnan(donchian_lower_12h_6h[i]) or 
            np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: weekly uptrend + price breaks above 12h Donchian upper + volume
            if (weekly_trend[i] == 1 and 
                close[i] > donchian_upper_12h_6h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below 12h Donchian lower + volume
            elif (weekly_trend[i] == -1 and 
                  close[i] < donchian_lower_12h_6h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: weekly trend reversal or price returns to opposite Donchian band
            if position == 1:
                # Exit long: weekly trend turns down OR price closes below 12h Donchian lower
                if weekly_trend[i] == -1 or close[i] < donchian_lower_12h_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns up OR price closes above 12h Donchian upper
                if weekly_trend[i] == 1 or close[i] > donchian_upper_12h_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals