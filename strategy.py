#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Uses 20-period Donchian channels from 1d data for trend direction and breakout levels.
# Filters by 1d ADX > 25 for trend strength and volume > 1.3x 20-period average.
# In trending markets (ADX>25): breakout continuation at Donchian breakout levels.
# In ranging markets (ADX<=25): no trades to avoid whipsaw.
# Designed to capture strong trends while avoiding choppy markets.
# Target: 12-37 trades/year via Donchian + trend + volume confluence on 12h timeframe.

name = "12h_donchian_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1d data
    # Upper channel = highest high over 20 periods
    # Lower channel = lowest low over 20 periods
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period ADX for trend strength on 1d data
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 60  # Need 20 for Donchian + 14 for ADX + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 12h bar
        high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)[i]
        low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)[i]
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)[i]
        
        # Regime filter: only trade in trending markets
        trending = adx_aligned > 25
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        if position == 1:  # Long position
            # Exit conditions: close below lower Donchian channel
            if close[i] < low_20_aligned:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: close above upper Donchian channel
            if close[i] > high_20_aligned:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation and in trending markets
            if volume_confirm and trending:
                # Breakout entry: long on break above upper channel, short on break below lower channel
                if close[i] > high_20_aligned:  # Bullish breakout
                    position = 1
                    signals[i] = 0.25
                elif close[i] < low_20_aligned:  # Bearish breakout
                    position = -1
                    signals[i] = -0.25
    
    return signals