#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_4H_1D_Donchian_20_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for Donchian channels (once before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on 4h and 1d
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # 4h ADX for trend strength (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(high_20_4h_aligned[i]) or np.isnan(low_20_4h_aligned[i]) or \
           np.isnan(high_20_1d_aligned[i]) or np.isnan(low_20_1d_aligned[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx[i]
        
        # Multi-timeframe Donchian conditions
        upper_4h = high_20_4h_aligned[i]
        lower_4h = low_20_4h_aligned[i]
        upper_1d = high_20_1d_aligned[i]
        lower_1d = low_20_1d_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        trending = adx_val > 22  # Strong trend filter
        
        if position == 0:
            # Long: Price breaks above both 4h and 1d upper bands + volume + trending
            if price > upper_4h and price > upper_1d and volume_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below both 4h and 1d lower bands + volume + trending
            elif price < lower_4h and price < lower_1d and volume_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 4h midpoint
            midpoint_4h = (upper_4h + lower_4h) / 2
            if price < midpoint_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above 4h midpoint
            midpoint_4h = (upper_4h + lower_4h) / 2
            if price > midpoint_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals