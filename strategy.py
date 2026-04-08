#!/usr/bin/env python3
"""
1h Breakout with 4h Trend Filter and Volume Spike v2
Hypothesis: 1h price breakouts from 20-period ranges, aligned with strong 4h trend (ADX>25) 
and volume confirmation (>2x average), capture momentum while avoiding false breakouts.
Uses 4h for trend direction, 1h for precise entry timing. Target: 15-30 trades/year.
Works in bull/bear by requiring trend alignment and volume confirmation.
"""

name = "1h_breakout_4h_trend_volume_v2"
timeframe = "1h"
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
    
    # Get 4h data for trend filter - call ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 14-period ADX for 4h
    # True Range
    tr1_4h = high_4h[1:] - low_4h[1:]
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    
    # Directional Movement
    dm_plus_4h = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                          np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus_4h = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                           np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus_4h = np.concatenate([[0], dm_plus_4h])
    dm_minus_4h = np.concatenate([[0], dm_minus_4h])
    
    # Smoothed values
    tr14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_4h = pd.Series(dm_plus_4h).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_4h = pd.Series(dm_minus_4h).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_4h = 100 * dm_plus_14_4h / tr14_4h
    di_minus_4h = 100 * dm_minus_14_4h / tr14_4h
    
    # DX and ADX
    dx_4h = 100 * np.abs(di_plus_4h - di_minus_4h) / (di_plus_4h + di_minus_4h)
    adx_4h = pd.Series(dx_4h).rolling(window=14, min_periods=14).mean().values
    
    # 1h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detector: current volume > 2 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_4h[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            not session_mask[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned 4h ADX for current 1h bar
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)[i]
        
        # Regime filter: only trade in strong trending markets on 4h
        strong_trend_4h = adx_4h_aligned > 25
        
        if position == 1:  # Long position
            # Exit: trend weakens OR price closes below Donchian low
            if not strong_trend_4h or close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend weakens OR price closes above Donchian high
            if not strong_trend_4h or close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only trade with volume spike and strong 4h trend
            # Breakout conditions: price breaks Donchian levels
            if volume_spike[i] and strong_trend_4h and close[i] > donch_high[i]:
                position = 1
                signals[i] = 0.20
            elif volume_spike[i] and strong_trend_4h and close[i] < donch_low[i]:
                position = -1
                signals[i] = -0.20
    
    return signals