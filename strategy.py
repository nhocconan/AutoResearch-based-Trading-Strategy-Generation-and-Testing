#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with volume spike and 1w ADX trend filter.
Long when price breaks above 20-day high AND volume > 2.0x 20-day average AND 1w ADX > 25.
Short when price breaks below 20-day low AND volume > 2.0x 20-day average AND 1w ADX > 25.
Exit when price reverts to 10-day MA OR 1w ADX < 20 (range market).
Uses 1d for price/volume, 1w for ADX filter to reduce whipsaw and capture major trends.
Target: 30-100 total trades over 4 years (7-25/year). Works in bull markets (captures uptrends) 
and bear markets (captures downtrends) by filtering with weekly trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels (based on previous 20 days, not including today)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 10-day MA for exit
    ma_10 = pd.Series(close_1d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w timeframe (14-period)
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
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / np.where(atr_1w != 0, atr_1w, np.inf))
    minus_di = 100 * (minus_dm_smooth / np.where(atr_1w != 0, atr_1w, np.inf))
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.inf)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    # Align 1w ADX to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        donchian_high = high_20[i]
        donchian_low = low_20[i]
        ma_10_val = ma_10[i]
        vol_ma = vol_ma_20[i]
        vol = volume_1d[i]
        price = close_1d[i]
        adx_val = adx_1w_aligned[i]
        
        if position == 0:
            # Long: price > 20-day high AND volume > 2.0x avg AND 1w ADX > 25 (trending)
            if price > donchian_high and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < 20-day low AND volume > 2.0x avg AND 1w ADX > 25 (trending)
            elif price < donchian_low and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < 10-day MA OR 1w ADX < 20 (range market)
            if price < ma_10_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > 10-day MA OR 1w ADX < 20 (range market)
            if price > ma_10_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_1wADX_Filter"
timeframe = "1d"
leverage = 1.0