#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_Volume_Regime_v1
Breakout above/below weekly Donchian(20) with volume confirmation and daily chop filter.
Long: price breaks above weekly Donchian high + volume > 1.5x average + chop < 61.8
Short: price breaks below weekly Donchian low + volume > 1.5x average + chop < 61.8
Exit when price crosses weekly Donchian midline or chop > 61.8 (range).
Designed to capture strong trends in both bull and bear markets with low frequency.
Target: 20-50 total trades over 4 years (5-12.5/year).
"""

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
    
    # === Weekly Donchian channels (20-period) ===
    df_1w = get_htf_data(prices, '1w')
    donch_high = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    
    # === Volume confirmation: 20-day average volume ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily Choppiness Index (14-period) ===
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    tr[0] = atr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((hh - ll) == 0, 50.0, chop)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: breakout above weekly Donchian high + volume > 1.5x avg + chop < 61.8 (trending)
            if (close[i] > donch_high_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakout below weekly Donchian low + volume > 1.5x avg + chop < 61.8 (trending)
            elif (close[i] < donch_low_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midline OR chop > 61.8 (range)
            if (close[i] < donch_mid_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midline OR chop > 61.8 (range)
            if (close[i] > donch_mid_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0