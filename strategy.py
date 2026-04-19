#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian_20_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and weekly pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channel (20-period) on 1d
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2.0
    
    # Calculate weekly pivot (using last 5 days)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align to 6h timeframe
    donch_high_6h = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_6h = align_htf_to_ltf(prices, df_1d, donch_low_20)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(donch_high_6h[i]) or np.isnan(donch_low_6h[i]) or np.isnan(weekly_pivot_6h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        donch_high = donch_high_6h[i]
        donch_low = donch_low_6h[i]
        weekly_pivot = weekly_pivot_6h[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above Donchian high AND above weekly pivot
            if price > donch_high and price > weekly_pivot and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND below weekly pivot
            elif price < donch_low and price < weekly_pivot and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below Donchian midpoint or weekly pivot
            if price < donch_mid_20[i] or price < weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above Donchian midpoint or weekly pivot
            if price > donch_mid_20[i] or price > weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: Donchian midpoint array needs to be aligned as well
    # Fix: Calculate and align Donchian midpoint
    donch_mid_20_6h = align_htf_to_ltf(prices, df_1d, donch_mid_20)