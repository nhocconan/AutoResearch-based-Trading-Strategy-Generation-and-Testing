#!/usr/bin/env python3
"""
1d_1w_Momentum_Volume_Regime
Strategy: Momentum breakout with volume confirmation and regime filter on daily timeframe.
Long: Price breaks above 20-day high + volume > 1.5x 20-day avg + weekly ADX > 25 (trending)
Short: Price breaks below 20-day low + volume > 1.5x 20-day avg + weekly ADX > 25 (trending)
Exit: Price returns to 20-day moving average
Position size: 0.25
Designed to capture momentum moves in trending markets while avoiding choppy conditions.
Timeframe: 1d
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
    
    # Calculate 20-day moving average for exit
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-day high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-day average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ADX
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need 20 for breakout, 34 for ADX (14+14+6)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Regime filter: weekly ADX > 25 (trending market)
        regime_filter = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_up = close[i] > high_20[i-1]  # break above previous 20-day high
        breakout_down = close[i] < low_20[i-1]  # break below previous 20-day low
        
        # Exit condition: return to 20-day moving average
        return_to_ma = abs(close[i] - ma20[i]) < 0.01 * close[i]  # within 1% of MA20
        
        if position == 0:
            # Long: breakout up + volume filter + regime filter
            if breakout_up and volume_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + volume filter + regime filter
            elif breakout_down and volume_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to MA20 or break down
            if return_to_ma or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to MA20 or break up
            if return_to_ma or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Momentum_Volume_Regime"
timeframe = "1d"
leverage = 1.0