#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d HTF data for weekly pivot (using 1d to calculate weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (using Friday's data)
    # We'll use the prior Friday's high/low/close to calculate weekly pivot
    # For simplicity, we'll use 1d data and calculate pivot from 5 days ago
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot using prior Friday's OHLC (5-day lookback for weekly)
    pivot_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values  # Weekly high
    pivot_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values    # Weekly low
    pivot_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values  # Weekly close approx
    
    # Weekly pivot point (standard calculation)
    weekly_pivot = (pivot_high + pivot_low + pivot_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - pivot_low
    weekly_s1 = 2 * weekly_pivot - pivot_high
    weekly_r2 = weekly_pivot + (pivot_high - pivot_low)
    weekly_s2 = weekly_pivot - (pivot_high - pivot_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Volume confirmation: 6h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Long conditions:
        # 1. Price breaks above 12h Donchian high (breakout)
        # 2. Price above weekly pivot (bullish bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i] and
            close[i] > weekly_pivot_aligned[i] and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 12h Donchian low (breakdown)
        # 2. Price below weekly pivot (bearish bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i] and
              close[i] < weekly_pivot_aligned[i] and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0