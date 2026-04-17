#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian(20) breakout + volume confirmation + ATR volatility filter.
Long when price breaks above 1d Donchian upper channel with volume confirmation and ATR ratio > 0.8 (sufficient volatility).
Short when price breaks below 1d Donchian lower channel with volume confirmation and ATR ratio > 0.8.
Exit when price returns to the 1d Donchian midpoint or ATR ratio drops below 0.3 (low volatility regime).
Uses 1d timeframe for structure (reduces noise) and 12h for entry/execution.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in low volatility markets.
Donchian channels provide dynamic support/resistance based on 20-period high/low, effective in both trending and ranging markets.
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
    
    # Get 1d data for Donchian calculation and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donchian_upper = high_1d_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_1d_series.rolling(window=20, min_periods=20).min().values
    donchian_midpoint = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio: current ATR / 50-period ATR average (volatility regime filter)
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50  # >1 = high volatility, <1 = low volatility
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_midpoint_aligned = align_htf_to_ltf(prices, df_1d, donchian_midpoint)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Donchian(20), ATR(14), ATR MA(50), and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_midpoint_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: ATR ratio > 0.8 (sufficient volatility for breakout)
        volatility_sufficient = atr_ratio_aligned[i] > 0.8
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume and sufficient volatility
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed and 
                volatility_sufficient):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume and sufficient volatility
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed and 
                  volatility_sufficient):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR volatility drops too low (ATR ratio < 0.3)
            if (close[i] <= donchian_midpoint_aligned[i] or 
                atr_ratio_aligned[i] < 0.3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR volatility drops too low (ATR ratio < 0.3)
            if (close[i] >= donchian_midpoint_aligned[i] or 
                atr_ratio_aligned[i] < 0.3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0