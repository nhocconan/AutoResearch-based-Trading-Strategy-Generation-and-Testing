#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1d ADX trend filter
# Long when price breaks above Donchian upper channel AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending market)
# Short when price breaks below Donchian lower channel AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending market)
# Exit when price crosses back to Donchian middle (20-period average) OR 1d ADX < 20 (range market)
# Uses discrete sizing (0.30) to limit fee drag. Target: 20-50 trades/year per symbol.
# Donchian channels provide structural breakouts, volume spike confirms conviction, 1d ADX filters for trending conditions to avoid chop.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends, avoids ranging markets.

name = "4h_Donchian20_VolumeSpike_1dADX_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h data (using current timeframe)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 14:
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        
        # Trend conditions
        trending = adx > 25
        ranging = adx < 20
    else:
        trending = np.full(len(df_1d), False)
        ranging = np.full(len(df_1d), False)
        adx = np.full(len(df_1d), np.nan)
    
    # Align 1d indicators to 4h timeframe
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    ranging_aligned = align_htf_to_ltf(prices, df_1d, ranging.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(trending_aligned[i]) or 
            np.isnan(ranging_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND 1d trending
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                trending_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND 1d trending
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  trending_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian mid OR 1d ranging
            if (close[i] < donchian_mid[i] or 
                ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back to Donchian mid OR 1d ranging
            if (close[i] > donchian_mid[i] or 
                ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals