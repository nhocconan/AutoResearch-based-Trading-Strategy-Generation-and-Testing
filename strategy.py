#!/usr/bin/env python3
"""
4h Triangular Moving Average (TMA) Slope with Volume Confirmation and 12h Trend Filter.
Trades in direction of TMA slope on 4h timeframe, confirmed by volume spikes,
only when 12h timeframe shows strong trend (ADX > 25). Designed for 4h timeframe
to target 75-200 total trades over 4 years (19-50/year).
Works in both bull and bear markets by trading with the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for TMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Triangular Moving Average (TMA) = SMA of SMA
    tma_period = 21
    sma1 = pd.Series(close_4h).rolling(window=tma_period, min_periods=tma_period).mean().values
    tma = pd.Series(sma1).rolling(window=tma_period, min_periods=tma_period).mean().values
    
    # TMA slope: positive = uptrend, negative = downtrend
    tma_slope = np.diff(tma, prepend=tma[0])
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr0 = np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
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
    
    # ADX > 25 = strong trend
    strong_trend = adx > 25
    
    # Volume spike: volume > 1.8x 20-period average on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.8)
    
    # Align indicators to lower timeframe (4h)
    tma_slope_aligned = align_htf_to_ltf(prices, df_4h, tma_slope)
    strong_trend_aligned = align_htf_to_ltf(prices, df_12h, strong_trend.astype(float))
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(tma_slope_aligned[i]) or 
            np.isnan(strong_trend_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: TMA slope direction + volume spike + strong trend
        long_entry = (tma_slope_aligned[i] > 0 and 
                      vol_spike_aligned[i] > 0.5 and 
                      strong_trend_aligned[i] > 0.5)
        short_entry = (tma_slope_aligned[i] < 0 and 
                       vol_spike_aligned[i] > 0.5 and 
                       strong_trend_aligned[i] > 0.5)
        
        # Exit when TMA slope changes sign
        exit_long = position == 1 and tma_slope_aligned[i] <= 0
        exit_short = position == -1 and tma_slope_aligned[i] >= 0
        
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

name = "4h_tma_slope_volume_trend"
timeframe = "4h"
leverage = 1.0