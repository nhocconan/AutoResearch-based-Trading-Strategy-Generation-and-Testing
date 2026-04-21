#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian channel breakout with 1d ADX trend filter and 1d volume confirmation.
The ADX > 25 confirms a trending market (bull or break) while avoiding ranging markets.
Breakouts in the direction of the trend capture momentum with reduced false signals.
Volume > 1.5x 20-period average confirms institutional participation.
Designed for ~15-25 trades/year to minimize fee drag, works in bull/bear via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ADX and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        adx_val = adx_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        
        # Calculate Donchian channel (20-period) using current close prices
        if i >= 20:
            high_20 = np.max(prices['high'].iloc[i-19:i+1].values)
            low_20 = np.min(prices['low'].iloc[i-19:i+1].values)
        else:
            # Not enough data for Donchian calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper, ADX > 25, volume confirmation
            if (price_close > high_20 and 
                adx_val > 25 and 
                vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower, ADX > 25, volume confirmation
            elif (price_close < low_20 and 
                  adx_val > 25 and 
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite Donchian band or ADX falls below 20
            if i >= 20:
                high_20_exit = np.max(prices['high'].iloc[i-19:i+1].values)
                low_20_exit = np.min(prices['low'].iloc[i-19:i+1].values)
            else:
                high_20_exit = high_20
                low_20_exit = low_20
            
            if position == 1 and (price_close < low_20_exit or adx_val < 20):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > high_20_exit or adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_DonchianBreakout_1dADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0