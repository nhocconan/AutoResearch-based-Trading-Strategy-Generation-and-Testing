#!/usr/bin/env python3
"""
6h Elder Ray Index with Volume Confirmation and Trend Filter.
Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with volume confirmation.
Short when Bear Power < 0 and falling, Bull Power < 0 and rising, with volume confirmation.
Uses 1-day EMA13 for Elder Ray calculation and 1-day ADX > 25 for trend filter.
Targets 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
Works in both bull and bear markets by trading with institutional power imbalances.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA13 for Elder Ray
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Components
    bull_power = high_1d - ema13  # High - EMA13
    bear_power = low_1d - ema13   # Low - EMA13
    
    # Slope of Elder Ray (3-period change)
    bull_power_slope = pd.Series(bull_power).diff(3).values
    bear_power_slope = pd.Series(bear_power).diff(3).values
    
    # Elder Ray Signals: Bullish when BP>0 and rising, Bearish when BP<0 and falling
    bullish_signal = (bull_power > 0) & (bull_power_slope > 0)
    bearish_signal = (bear_power < 0) & (bear_power_slope < 0)
    
    # Align Elder Ray signals to 6h timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_signal.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_signal.astype(float))
    
    # Volume confirmation: volume > 1.3x 20-period average (1d)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_1d > (vol_ma_20 * 1.3)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm.astype(float))
    
    # 1-day ADX (14-period) for trend filter
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr0 = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX > 25 = trending market
    trending = adx > 25
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i]) or 
            np.isnan(trending_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = bullish_aligned[i] > 0.5 and vol_confirm_aligned[i] > 0.5 and trending_aligned[i] > 0.5
        short_entry = bearish_aligned[i] > 0.5 and vol_confirm_aligned[i] > 0.5 and trending_aligned[i] > 0.5
        
        # Exit when Elder Ray signal reverses
        exit_long = position == 1 and bearish_aligned[i] > 0.5
        exit_short = position == -1 and bullish_aligned[i] > 0.5
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_elder_ray_volume_trend"
timeframe = "6h"
leverage = 1.0