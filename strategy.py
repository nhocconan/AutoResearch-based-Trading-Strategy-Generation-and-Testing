#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour volume confirmation and 1-day ADX trend filter
# Long when price breaks above 4h Donchian upper band with volume >1.3x 20-period average and 12h ADX >25
# Short when price breaks below 4h Donchian lower band with volume >1.3x 20-period average and 12h ADX >25
# Exit when price crosses the 4h Donchian midline
# 12h ADX acts as trend strength filter to avoid ranging markets and reduce whipsaws
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h, 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h Donchian channel (20-period lookback)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 12h ADX (14-period) for trend strength
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])) > 
                       (np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h), 
                       np.maximum(high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h) > 
                        (high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])), 
                        np.maximum(np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h, 0), 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 20-period + 14-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume confirmation and strong trend
            if (price > donchian_upper_aligned[i] and 
                vol_4h_current > 1.3 * vol_ma_4h_aligned[i] and  # Volume confirmation
                adx_aligned[i] > 25):                           # Strong trend filter
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume confirmation and strong trend
            elif (price < donchian_lower_aligned[i] and 
                  vol_4h_current > 1.3 * vol_ma_4h_aligned[i] and  # Volume confirmation
                  adx_aligned[i] > 25):                           # Strong trend filter
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_12hADX_Volume"
timeframe = "4h"
leverage = 1.0