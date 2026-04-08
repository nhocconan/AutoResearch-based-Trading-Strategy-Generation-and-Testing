#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Spike and Weekly ADX Filter v1
Hypothesis: Donchian(20) breakouts on 12h with volume spikes (>2x average) 
and strong weekly trend (ADX > 25) capture sustained moves while avoiding 
false breakouts in ranging markets. Works in bull/bear by requiring 
trend alignment and volume confirmation. Target: 12-37 trades/year.
"""

name = "12h_donchian_breakout_volume_adx_v1"
timeframe = "12h"
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
    
    # Get weekly data for trend filter - call ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 14-period ADX for weekly
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
    
    # Smoothed values
    tr14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_1w = pd.Series(dm_plus_1w).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_1w = pd.Series(dm_minus_1w).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1w = 100 * dm_plus_14_1w / tr14_1w
    di_minus_1w = 100 * dm_minus_14_1w / tr14_1w
    
    # DX and ADX
    dx_1w = 100 * np.abs(di_plus_1w - di_minus_1w) / (di_plus_1w + di_minus_1w)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels on 12h (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detector: current volume > 2 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1w[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned weekly ADX for current 12h bar
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)[i]
        
        # Regime filter: only trade in strong trending markets on weekly
        strong_trend_1w = adx_1w_aligned > 25
        
        if position == 1:  # Long position
            # Exit: trend weakens OR price closes below Donchian low
            if not strong_trend_1w or close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakens OR price closes above Donchian high
            if not strong_trend_1w or close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume spike and strong weekly trend
            # Breakout conditions: price breaks Donchian levels
            if volume_spike[i] and strong_trend_1w and close[i] > donch_high[i]:
                position = 1
                signals[i] = 0.25
            elif volume_spike[i] and strong_trend_1w and close[i] < donch_low[i]:
                position = -1
                signals[i] = -0.25
    
    return signals