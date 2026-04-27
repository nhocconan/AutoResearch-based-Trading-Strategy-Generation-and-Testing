#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with Volume Confirmation and ADX Trend Filter
Long when price breaks above Donchian upper band (20-period high) + volume > 1.5x average + ADX > 25
Short when price breaks below Donchian lower band (20-period low) + volume > 1.5x average + ADX > 25
Exit when price crosses back below the 20-period SMA (long) or above the 20-period SMA (short)
Designed to generate 10-25 trades/year per symbol with strong trend-following edge in both bull and bear markets
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
    
    # Get 1d data for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channels: 20-period high and low
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # ADX calculation (14-period)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align to 1d timeframe (already in 1d, but using align_htf_to_ltf for proper delay)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # 20-period SMA for exit
    sma_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Volume filter: volume > 1.5x average (20-period)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + ADX (14+14) + volume MA (20)
    start_idx = max(20, 28, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(sma_20_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicator values
        upper_band = high_20_aligned[i]
        lower_band = low_20_aligned[i]
        adx_val = adx_aligned[i]
        sma_val = sma_20_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        # ADX filter: strong trend
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long breakout: price > upper band + volume + trend
            if price_now > upper_band and vol_filter and trend_filter:
                signals[i] = size
                position = 1
            # Short breakdown: price < lower band + volume + trend
            elif price_now < lower_band and vol_filter and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below 20-period SMA
            if price_now < sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above 20-period SMA
            if price_now > sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_20_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0