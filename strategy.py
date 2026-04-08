#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Price breaks out of 20-period Donchian channel on 4h chart when 
daily trend is strong (ADX>25) and volume is above average, capturing momentum 
in both bull and bear markets. Uses volume filter to avoid false breakouts.
Target: 20-40 trades/year per symbol.
"""

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter - call ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period ADX for 1d
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_1d = np.concatenate([[0], dm_plus_1d])
    dm_minus_1d = np.concatenate([[0], dm_minus_1d])
    
    # Smoothed values
    tr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_1d = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_1d = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1d = 100 * dm_plus_14_1d / tr14_1d
    di_minus_1d = 100 * dm_minus_14_1d / tr14_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume average for 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(adx_1d[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d ADX for current 4h bar
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Regime detection: only trade in strong trends
        strong_trend_1d = adx_1d_aligned > 25
        
        # Volume confirmation: current volume above 20-period average
        volume_confirm = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on Donchian lower band break (stop loss)
            if i >= 20:
                donchian_low = np.min(low[i-20:i])
                if close[i] < donchian_low:
                    position = 0
                    signals[i] = 0.0
            if position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on Donchian upper band break (stop loss)
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                if close[i] > donchian_high:
                    position = 0
                    signals[i] = 0.0
            if position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if i < 20:
                signals[i] = 0.0
                continue
                
            # Calculate Donchian channels (20-period)
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
            
            # Entry logic: breakout with volume confirmation in strong trend
            if strong_trend_1d and volume_confirm:
                if close[i] > donchian_high and close[i-1] <= donchian_high:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donchian_low and close[i-1] >= donchian_low:
                    position = -1
                    signals[i] = -0.25
    
    return signals