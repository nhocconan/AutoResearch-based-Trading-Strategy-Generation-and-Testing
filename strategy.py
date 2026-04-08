#!/usr/bin/env python3
"""
12h Donchian(15) breakout with 1w ADX trend filter and volume confirmation
Hypothesis: Price breaking above/below 12-hour Donchian channels during strong weekly trends
(captured by ADX > 25 on weekly) with volume confirmation captures sustained moves while
avoiding whipsaw. Using weekly trend filter reduces trade frequency to target 12-37 trades/year.
Works in both bull and bear markets by requiring strong weekly trend alignment.
"""

name = "12h_donchian_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX for 1w data
    # True Range
    tr1_1w = high_1w[1:] - low_1w[1:]
    tr2_1w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_1w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))])
    
    # Directional Movement
    dm_plus_1w = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                          np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus_1w = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                           np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus_1w = np.concatenate([[0], dm_plus_1w])
    dm_minus_1w = np.concatenate([[0], dm_minus_1w])
    
    # Smoothed values with proper min_periods
    tr14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_1w = pd.Series(dm_plus_1w).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_1w = pd.Series(dm_minus_1w).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1w = 100 * dm_plus_14_1w / tr14_1w
    di_minus_1w = 100 * dm_minus_14_1w / tr14_1w
    
    # DX and ADX
    dx_1w = 100 * np.abs(di_plus_1w - di_minus_1w) / (di_plus_1w + di_minus_1w)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    
    # 15-period volume average for confirmation
    vol_avg_15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    
    # 15-period Donchian channels for breakout signals on 12h data
    donchian_high = pd.Series(high).rolling(window=15, min_periods=15).max().values
    donchian_low = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30  # Need ADX and Donchian buffers
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1w[i]) or 
            np.isnan(vol_avg_15[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1w value for current 12h bar
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)[i]
        
        # Regime filter: only trade in strong trending markets on weekly
        strong_trend_1w = adx_1w_aligned > 25
        
        # Volume confirmation: current volume > 1.3x 15-period average
        volume_confirm = volume[i] > 1.3 * vol_avg_15[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 15-period Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 15-period Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation and in strong trending markets on weekly
            if volume_confirm and strong_trend_1w:
                # Long entry: price breaks above Donchian high
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals