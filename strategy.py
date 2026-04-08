#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and 12h ADX Filter v2
Hypothesis: Donchian(20) breakouts on 4h with volume spikes (>2x average) 
and strong 12h trend (ADX > 25) capture sustained moves while avoiding 
false breakouts in ranging markets. Works in bull/bear by requiring 
trend alignment and volume confirmation. Target: 20-50 trades/year.
"""

name = "4h_donchian_breakout_volume_adx_v2"
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
    
    # Get 12h data for trend filter - call ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 14-period ADX for 12h
    # True Range
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    
    # Directional Movement
    dm_plus_12h = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                           np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus_12h = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                            np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus_12h = np.concatenate([[0], dm_plus_12h])
    dm_minus_12h = np.concatenate([[0], dm_minus_12h])
    
    # Smoothed values
    tr14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_12h = pd.Series(dm_plus_12h).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_12h = pd.Series(dm_minus_12h).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_12h = 100 * dm_plus_14_12h / tr14_12h
    di_minus_12h = 100 * dm_minus_14_12h / tr14_12h
    
    # DX and ADX
    dx_12h = 100 * np.abs(di_plus_12h - di_minus_12h) / (di_plus_12h + di_minus_12h)
    adx_12h = pd.Series(dx_12h).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels on 4h (20-period)
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
        if (np.isnan(adx_12h[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 12h ADX for current 4h bar
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)[i]
        
        # Regime filter: only trade in strong trending markets on 12h
        strong_trend_12h = adx_12h_aligned > 25
        
        if position == 1:  # Long position
            # Exit: trend weakens OR price closes below Donchian low
            if not strong_trend_12h or close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakens OR price closes above Donchian high
            if not strong_trend_12h or close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume spike and strong 12h trend
            # Breakout conditions: price breaks Donchian levels
            if volume_spike[i] and strong_trend_12h and close[i] > donch_high[i]:
                position = 1
                signals[i] = 0.25
            elif volume_spike[i] and strong_trend_12h and close[i] < donch_low[i]:
                position = -1
                signals[i] = -0.25
    
    return signals