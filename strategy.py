#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and ADX trend filter
# Works in both bull and breakout markets: breakouts capture strong moves, volume confirms institutional interest, ADX avoids false signals in ranging markets
# Designed for low trade frequency (target 20-40/year) with clear breakout logic and built-in risk control via position sizing

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h - using previous close to avoid look-ahead
    # We use the 4h bar that closed at least one period ago
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 12h volume average (20-period)
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX (14-period) on 1d for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1d[1:]]) - high_1d[:-1]
    down_move = low_1d[:-1] - np.concatenate([[np.nan], low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to the main price timeframe (assumed to be 4h based on strategy design)
    # Since we're using 4h as primary, we need to align 12h and 1d data to 4h
    donchian_high_aligned = align_htf_to_ltf(high_4h, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(low_4h, df_4h, donchian_low)
    vol_avg_12h_aligned = align_htf_to_ltf(volume, df_12h, vol_avg_12h)
    adx_aligned = align_htf_to_ltf(close, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size - 25% of capital
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + strong trend (ADX > 25)
        if (close[i] > donchian_high_aligned[i] and 
            volume[i] > 2.0 * vol_avg_12h_aligned[i] and 
            adx_aligned[i] > 25 and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + strong trend (ADX > 25)
        elif (close[i] < donchian_low_aligned[i] and 
              volume[i] > 2.0 * vol_avg_12h_aligned[i] and 
              adx_aligned[i] > 25 and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or trend weakness (ADX < 20) to avoid whipsaws
        elif position == 1 and (close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_12hVolume_ADX_Filter"
timeframe = "4h"
leverage = 1.0