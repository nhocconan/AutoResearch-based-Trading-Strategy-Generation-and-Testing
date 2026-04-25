#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h ADX Trend + Volume Confirmation
Hypothesis: Donchian breakouts capture momentum moves, filtered by 12h ADX>25 for trending regimes and volume confirmation for conviction. Works in both bull and bear markets by taking breakouts in direction of 12h trend. Targets 50-150 trades over 4 years with discrete sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=np.float64)
    tr1 = pd.Series(high).values - pd.Series(low).values
    tr2 = np.abs(pd.Series(high).values - pd.Series(close).shift(1).values)
    tr3 = np.abs(pd.Series(low).values - pd.Series(close).shift(1).values)
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).values
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=np.float64), np.full_like(high, np.nan, dtype=np.float64), np.full_like(high, np.nan, dtype=np.float64)
    
    # True Range
    tr1 = pd.Series(high).values - pd.Series(low).values
    tr2 = np.abs(pd.Series(high).values - pd.Series(close).shift(1).values)
    tr3 = np.abs(pd.Series(low).values - pd.Series(close).shift(1).values)
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).values
    
    # Directional Movement
    dm_plus = np.where((pd.Series(high).values - pd.Series(high).shift(1).values) > 
                       (pd.Series(low).shift(1).values - pd.Series(low).values),
                       np.maximum(pd.Series(high).values - pd.Series(high).shift(1).values, 0), 0)
    dm_minus = np.where((pd.Series(low).shift(1).values - pd.Series(low).values) > 
                        (pd.Series(high).values - pd.Series(high).shift(1).values),
                        np.maximum(pd.Series(low).shift(1).values - pd.Series(low).values, 0), 0)
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (np.abs(di_plus) + np.abs(di_minus)) * 100
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, di_plus, di_minus

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 12h data
    adx_12h, di_plus_12h, di_minus_12h = calculate_adx(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        14
    )
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    di_plus_12h_aligned = align_htf_to_ltf(prices, df_12h, di_plus_12h)
    di_minus_12h_aligned = align_htf_to_ltf(prices, df_12h, di_minus_12h)
    
    # Calculate Donchian channels (20-period) on 6h data
    donchian_upper = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, ADX, volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(di_plus_12h_aligned[i]) or
            np.isnan(di_minus_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        adx_val = adx_12h_aligned[i]
        di_plus_val = di_plus_12h_aligned[i]
        di_minus_val = di_minus_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Determine 12h trend direction
        uptrend_12h = (adx_val > 25) and (di_plus_val > di_minus_val)
        downtrend_12h = (adx_val > 25) and (di_minus_val > di_plus_val)
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper AND volume confirmation AND 12h uptrend
            long_entry = (curr_close > donchian_upper[i]) and vol_conf and uptrend_12h
            # Short: price breaks below Donchian lower AND volume confirmation AND 12h downtrend
            short_entry = (curr_close < donchian_lower[i]) and vol_conf and downtrend_12h
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Donchian lower OR 12h trend changes to downtrend
            if (curr_close < donchian_lower[i]) or (not uptrend_12h and adx_val > 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian upper OR 12h trend changes to uptrend
            if (curr_close > donchian_upper[i]) or (not downtrend_12h and adx_val > 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hADX_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0