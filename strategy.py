#!/usr/bin/env python3
"""
6h Elder Ray + Volume + 1-day Trend Filter
Trades Elder Ray bull/bear power crossovers with volume confirmation, only when 1-day ADX > 25.
Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by trading with the trend as measured by daily ADX.
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
    
    # Get 6h data for EMA13 (Elder Ray)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_6h - ema13
    bear_power = low_6h - ema13
    
    # Elder Ray signals: bull power crosses above zero (long), bear power crosses below zero (short)
    bull_cross_up = (bull_power[1:] > 0) & (bull_power[:-1] <= 0)
    bear_cross_down = (bear_power[1:] < 0) & (bear_power[:-1] >= 0)
    bull_cross_up = np.concatenate([[False], bull_cross_up])
    bear_cross_down = np.concatenate([[False], bear_cross_down])
    
    # Align Elder Ray signals to 6h
    bull_cross_up_aligned = align_htf_to_ltf(prices, df_6h, bull_cross_up.astype(float))
    bear_cross_down_aligned = align_htf_to_ltf(prices, df_6h, bear_cross_down.astype(float))
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
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
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX > 25 = trending market
    trending = adx > 25
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_cross_up_aligned[i]) or 
            np.isnan(bear_cross_down_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(trending_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Elder Ray crossover + volume spike + trending market
        long_entry = (bull_cross_up_aligned[i] > 0.5 and 
                      vol_spike_aligned[i] > 0.5 and 
                      trending_aligned[i] > 0.5)
        short_entry = (bear_cross_down_aligned[i] > 0.5 and 
                       vol_spike_aligned[i] > 0.5 and 
                       trending_aligned[i] > 0.5)
        
        # Exit when opposite Elder Ray signal occurs
        exit_long = position == 1 and bear_cross_down_aligned[i] > 0.5
        exit_short = position == -1 and bull_cross_up_aligned[i] > 0.5
        
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