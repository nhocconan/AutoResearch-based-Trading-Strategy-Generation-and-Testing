#!/usr/bin/env python3
"""
4h Camarilla Pivot Reversal with Volume Spike and 1-day ADX Trend Filter.
Trades reversals at Camarilla pivot levels (S3/S4 for long, R3/R4 for short) 
confirmed by volume spikes, only in trending markets (1-day ADX > 25) to avoid 
false signals in ranging conditions. Designed for 4h timeframe to target 75-200 
total trades over 4 years (19-50/year). Works in both bull and bear markets by 
trading reversals in the direction of the trend.
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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # Camarilla formulas: 
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # We'll use H3/L3 and H4/L4 for entries
    
    # Calculate daily range from previous day
    prev_close = np.concatenate([[close_4h[0]], close_4h[:-1]])
    prev_high = np.concatenate([[high_4h[0]], high_4h[:-1]])
    prev_low = np.concatenate([[low_4h[0]], low_4h[:-1]])
    
    # Camarilla levels
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.1 * (prev_high - prev_low)
    L3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Reversal conditions: price touches H3/L4 (short) or L3/H4 (long) with rejection
    # We'll use touches as entry signals
    touch_H3 = (high_4h >= H3) & (close_4h < H3)  # touched H3 but closed below
    touch_L3 = (low_4h <= L3) & (close_4h > L3)   # touched L3 but closed above
    touch_H4 = (high_4h >= H4) & (close_4h < H4)  # touched H4 but closed below
    touch_L4 = (low_4h <= L4) & (close_4h > L4)   # touched L4 but closed above
    
    # Short signals: touch resistance levels
    short_signal = touch_H3 | touch_H4
    # Long signals: touch support levels  
    long_signal = touch_L3 | touch_L4
    
    # Align signals to 4h timeframe
    short_signal_aligned = align_htf_to_ltf(prices, df_4h, short_signal.astype(float))
    long_signal_aligned = align_htf_to_ltf(prices, df_4h, long_signal.astype(float))
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Volume spike: volume > 2.0x 20-period average (strong confirmation)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
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
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) > 0, dx, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX > 25 = trending market (good for reversals with trend)
    trending = adx > 25
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(long_signal_aligned[i]) or 
            np.isnan(short_signal_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(trending_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla touch + volume spike + trending market
        long_entry = (long_signal_aligned[i] > 0.5 and 
                      vol_spike_aligned[i] > 0.5 and 
                      trending_aligned[i] > 0.5)
        short_entry = (short_signal_aligned[i] > 0.5 and 
                       vol_spike_aligned[i] > 0.5 and 
                       trending_aligned[i] > 0.5)
        
        # Exit when price reaches opposite Camarilla level or midpoint
        mid_point = (H3 + L3) / 2
        mid_point_aligned = align_htf_to_ltf(prices, df_4h, np.full_like(close_4h, mid_point))
        
        exit_long = position == 1 and close[i] <= mid_point_aligned[i]
        exit_short = position == -1 and close[i] >= mid_point_aligned[i]
        
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

name = "4h_camarilla_pivot_reversal"
timeframe = "4h"
leverage = 1.0