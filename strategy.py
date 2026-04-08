#!/usr/bin/env python3
"""
1d Bollinger Bands with Weekly Trend Filter and Volume Confirmation v1
Hypothesis: Price reverses from Bollinger Band extremes on daily timeframe 
when aligned with weekly trend (ADX>25) and volume spikes (>1.5x 20-day average).
Works in bull/bear markets by requiring trend alignment. Target: 15-25 trades/year per symbol.
"""

name = "1d_bollinger_weekly_trend_volume_v1"
timeframe = "1d"
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
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate 14-period ADX for weekly
    # True Range
    tr1_weekly = high_weekly[1:] - low_weekly[1:]
    tr2_weekly = np.abs(high_weekly[1:] - close_weekly[:-1])
    tr3_weekly = np.abs(low_weekly[1:] - close_weekly[:-1])
    tr_weekly = np.concatenate([[np.nan], np.maximum(tr1_weekly, np.maximum(tr2_weekly, tr3_weekly))])
    
    # Directional Movement
    dm_plus_weekly = np.where((high_weekly[1:] - high_weekly[:-1]) > (low_weekly[:-1] - low_weekly[1:]), 
                              np.maximum(high_weekly[1:] - high_weekly[:-1], 0), 0)
    dm_minus_weekly = np.where((low_weekly[:-1] - low_weekly[1:]) > (high_weekly[1:] - high_weekly[:-1]), 
                               np.maximum(low_weekly[:-1] - low_weekly[1:], 0), 0)
    dm_plus_weekly = np.concatenate([[0], dm_plus_weekly])
    dm_minus_weekly = np.concatenate([[0], dm_minus_weekly])
    
    # Smoothed values
    tr14_weekly = pd.Series(tr_weekly).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_weekly = pd.Series(dm_plus_weekly).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_weekly = pd.Series(dm_minus_weekly).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_weekly = 100 * dm_plus_14_weekly / tr14_weekly
    di_minus_weekly = 100 * dm_minus_14_weekly / tr14_weekly
    
    # DX and ADX
    dx_weekly = 100 * np.abs(di_plus_weekly - di_minus_weekly) / (di_plus_weekly + di_minus_weekly)
    adx_weekly = pd.Series(dx_weekly).rolling(window=14, min_periods=14).mean().values
    
    # Daily Bollinger Bands (20-period, 2 std dev)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Volume spike detector: current volume > 1.5 x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_weekly[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned weekly ADX for current daily bar
        adx_weekly_aligned = align_htf_to_ltf(prices, df_weekly, adx_weekly)[i]
        
        # Regime filter: only trade in strong trending markets on weekly
        strong_trend_weekly = adx_weekly_aligned > 25
        
        if position == 1:  # Long position
            # Exit: trend weakens OR price touches middle band
            if not strong_trend_weekly or close[i] >= sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakens OR price touches middle band
            if not strong_trend_weekly or close[i] <= sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume spike and strong weekly trend
            # Mean reversion: price touches Bollinger Bands
            if volume_spike[i] and strong_trend_weekly and close[i] <= bb_lower[i]:
                position = 1
                signals[i] = 0.25
            elif volume_spike[i] and strong_trend_weekly and close[i] >= bb_upper[i]:
                position = -1
                signals[i] = -0.25
    
    return signals