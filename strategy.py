#!/usr/bin/env python3
# 1d Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# Hypothesis: Price breaking above/below daily Donchian channels during strong weekly trends
# (captured by ADX > 25 on weekly) with volume capture sustained moves while avoiding whipsaw.
# Weekly trend filter reduces trade frequency to target 7-25 trades/year. Works in bull/bear markets.

name = "1d_donchian_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for ADX (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX for weekly data
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
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 20-period Donchian channels for breakout signals on daily data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 40  # Need ADX and Donchian buffers
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1w[i]) or 
            np.isnan(vol_avg_20[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned weekly value for current daily bar
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)[i]
        
        # Regime filter: only trade in strong trending markets on weekly
        strong_trend_1w = adx_1w_aligned > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 20-period Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 20-period Donchian high
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