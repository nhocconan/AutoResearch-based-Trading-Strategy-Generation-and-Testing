#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w ADX trend filter.
Long when price breaks above Donchian upper band AND volume > 1.5x average AND ADX > 25 (trending).
Short when price breaks below Donchian lower band AND volume > 1.5x average AND ADX > 25.
Exit when price reverts to Donchian middle (20-period mean) OR ADX < 20 (range market).
Uses 1d for Donchian calculation and 1w for ADX filter to reduce whipsaw and capture major trends.
Target: 30-100 total trades over 4 years (7-25/year). Donchian breakouts capture strong trends,
volume confirmation filters fakeouts, weekly ADX filter avoids ranging markets and false breakouts.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
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
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels on 1d timeframe (20-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donchian_upper = high_1d_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_1d_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((donchian_upper + donchian_lower) / 2).values
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w timeframe (14-period)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / np.where(atr != 0, atr, np.inf))
    minus_di = 100 * (minus_dm_smooth / np.where(atr != 0, atr, np.inf))
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.inf)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d Donchian to 1d timeframe (no alignment needed)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    donchian_middle_aligned = donchian_middle
    
    # Align 1w ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume average (20-period) on 1d
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        du = donchian_upper_aligned[i]
        dl = donchian_lower_aligned[i]
        dm = donchian_middle_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 1.5x avg AND ADX > 25 (trending)
            if price > du and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume > 1.5x avg AND ADX > 25 (trending)
            elif price < dl and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle OR ADX < 20 (range market)
            if price < dm or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle OR ADX < 20 (range market)
            if price > dm or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_1wADX_Filter"
timeframe = "1d"
leverage = 1.0