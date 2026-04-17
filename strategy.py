#!/usr/bin/env python3
"""
12h Donchian Breakout + 1d Volume Spike + ADX Trend Filter
Long: Price breaks above Donchian(20) high + 1d volume > 1.5x 10-day avg + ADX(1d) > 25
Short: Price breaks below Donchian(20) low + 1d volume > 1.5x 10-day avg + ADX(1d) > 25
Exit: Opposite Donchian breakout or ADX < 20
Designed to capture strong trends with volume confirmation and trend strength filter.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 10-day average volume for 1d
    vol_avg_10d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate ADX(14) on 1d
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di_14 = np.where(tr_14 != 0, 100 * plus_dm_14 / tr_14, 0)
    minus_di_14 = np.where(tr_14 != 0, 100 * minus_dm_14 / tr_14, 0)
    
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 0)
    adx_14 = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 12h timeframe
    vol_avg_10d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10d)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Donchian channels (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 30)  # need Donchian and 1d indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_avg_10d_aligned[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d = volume_1d[i // 12] if i >= 12 else volume_1d[0]  # approximate 1d volume from 12h data
        vol_avg_val = vol_avg_10d_aligned[i]
        adx_val = adx_14_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        
        if position == 0:
            # Long: Price above Donchian high + volume spike + strong trend
            if price > donch_high and vol_1d > 1.5 * vol_avg_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price below Donchian low + volume spike + strong trend
            elif price < donch_low and vol_1d > 1.5 * vol_avg_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price below Donchian low or weak trend (ADX < 20)
            if price < donch_low or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price above Donchian high or weak trend (ADX < 20)
            if price > donch_high or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0