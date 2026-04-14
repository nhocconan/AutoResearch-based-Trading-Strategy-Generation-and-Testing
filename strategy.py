#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Donchian channel breakout with volume confirmation and ADX trend filter.
# Long when price breaks above 1-day Donchian upper channel (20-period high) AND volume > 1.5x 20-period average volume AND ADX > 25.
# Short when price breaks below 1-day Donchian lower channel (20-period low) AND volume > 1.5x 20-period average volume AND ADX > 25.
# Exit when price returns to the 1-day Donchian middle (midpoint of upper/lower) OR ADX drops below 20.
# Donchian channels provide clear trend-following structure; volume confirmation avoids false breakouts; ADX filter ensures trading in trending markets.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag and improve generalization.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels, volume average, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need enough for Donchian(20) and ADX(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day Donchian channels (20-period)
    # Upper channel: highest high of last 20 days
    # Lower channel: lowest low of last 20 days
    # Middle channel: midpoint of upper and lower
    lookback = 20
    
    # For Donchian, we need to use rolling max/min with proper min_periods
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    
    donchian_upper = high_series.rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = low_series.rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 20-period average volume for volume confirmation
    volume_series = pd.Series(volume_1d)
    avg_volume = volume_series.rolling(window=lookback, min_periods=lookback).mean().values
    
    # Calculate ADX (14) for trend strength
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:14])  # First ATR: simple average of first 14 TR
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values for DI calculation
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need Donchian and ADX)
    start = max(lookback, 27)  # Donchian(20) needs 20, ADX needs ~27 for full calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(avg_volume_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > (1.5 * avg_volume_aligned[i])
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for breakout entries in strong trend with volume confirmation
            # Long: price breaks above Donchian upper channel AND volume confirmation AND strong trend
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirm and 
                strong_trend):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower channel AND volume confirmation AND strong trend
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirm and 
                  strong_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian middle or trend weakens
            if (close[i] <= donchian_middle_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian middle or trend weakens
            if (close[i] >= donchian_middle_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0