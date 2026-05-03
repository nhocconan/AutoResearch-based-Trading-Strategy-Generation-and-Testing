#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ADX regime filter + volume spike confirmation.
# Donchian breakout captures sustained moves in both bull and bear markets.
# 1d ADX > 25 filters for trending regimes to avoid whipsaws in ranging markets.
# Volume spike (volume > 1.5 * 20-period MA) confirms breakout strength.
# Designed to capture sustained moves with tight entries to minimize fee drag.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_Donchian20_1dADX_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for regime filtering
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.concatenate([[np.nan], np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                                                 np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)])
    dm_minus = np.concatenate([[np.nan], np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                                                  np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smoothed = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smoothed = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di_1d = 100 * dm_plus_smoothed / atr_1d
    minus_di_1d = 100 * dm_minus_smoothed / atr_1d
    
    # Directional Index (DX) and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    # Handle division by zero when both DI are zero
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 12h volume spike confirmation (volume > 1.5 * 20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Regime filter: only trade when ADX > 25 (trending market)
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian upper channel AND volume spike AND trending regime AND session
            if close[i] > highest_high[i] and volume_spike[i] and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower channel AND volume spike AND trending regime AND session
            elif close[i] < lowest_low[i] and volume_spike[i] and is_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower channel OR reverse signal
            if close[i] < lowest_low[i] or (close[i] < lowest_low[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper channel OR reverse signal
            if close[i] > highest_high[i] or (close[i] > highest_high[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals