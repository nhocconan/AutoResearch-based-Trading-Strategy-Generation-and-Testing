#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot bias and volume confirmation.
Long when price breaks above 6h Donchian high AND weekly pivot bias bullish (price > weekly PP) AND volume > 1.5x 20-period average.
Short when price breaks below 6h Donchian low AND weekly pivot bias bearish (price < weekly PP) AND volume > 1.5x 20-period average.
Exit when price crosses the 6h Donchian midpoint (mean of high/low channel).
Weekly pivot from 1d timeframe provides structural bias, Donchian breakout captures momentum,
volume confirmation reduces false signals. Designed to work in both bull and bear markets
by using weekly pivot for regime and Donchian for entries/exits.
Targets 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) on 6h timeframe: upper = max(high,20), lower = min(low,20)
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    donchian_high = high_6h_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_6h_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate weekly pivot from 1d timeframe (using prior week's HLC)
    # Weekly PP = (Prior week HIGH + LOW + CLOSE) / 3
    # We need to get weekly values from daily data - use rolling window of 5 days
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    weekly_high = high_1d_series.rolling(window=5, min_periods=5).max().shift(1)  # prior week
    weekly_low = low_1d_series.rolling(window=5, min_periods=5).min().shift(1)   # prior week
    weekly_close = close_1d_series.rolling(window=5, min_periods=5).last().shift(1) # prior week
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Calculate volume average (20-period) on 6h
    volume_6h_series = pd.Series(df_6h['volume'].values)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for Donchian(20) and weekly pivot
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        mid = donchian_mid_aligned[i]
        weekly_pp = weekly_pp_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly PP bullish (price > weekly PP) AND volume > 1.5x avg
            if high_price > upper and price > weekly_pp and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND weekly PP bearish (price < weekly PP) AND volume > 1.5x avg
            elif low_price < lower and price < weekly_pp and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0