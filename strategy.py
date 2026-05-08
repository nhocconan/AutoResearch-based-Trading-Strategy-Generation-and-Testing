#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h volume spike and 1d ADX trend filter.
# Long when price breaks above Camarilla R3 AND 12h volume > 1.5x 20-period average AND ADX(14) > 25 (trending).
# Short when price breaks below Camarilla S3 AND 12h volume > 1.5x 20-period average AND ADX(14) > 25.
# Exit when price crosses back inside Camarilla H-L range.
# Uses 4h timeframe with 12h volume and 1d ADX for higher timeframe context.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled frequency to avoid fee drag.

name = "4h_Camarilla_R3S3_12hVolume_1dADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Camarilla calculation and ADX
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R3 = close + 1.1*(high-low)/1.1*6 = close + (high-low)/2
    # S3 = close - 1.1*(high-low)/1.1*6 = close - (high-low)/2
    prev_close_d = df_d['close'].values
    prev_high_d = df_d['high'].values
    prev_low_d = df_d['low'].values
    
    camarilla_r3_d = prev_close_d + 0.5 * (prev_high_d - prev_low_d)
    camarilla_s3_d = prev_close_d - 0.5 * (prev_high_d - prev_low_d)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r3 = align_htf_to_ltf(prices, df_d, camarilla_r3_d)
    camarilla_s3 = align_htf_to_ltf(prices, df_d, camarilla_s3_d)
    
    # 12h volume filter: current volume > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    vol_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma20_12h)
    volume_filter = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    # 1d ADX(14) for trend strength
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]  # First TR
    
    # Directional Movement
    plus_dm = np.where((high_d - np.roll(high_d, 1)) > (np.roll(low_d, 1) - low_d), 
                       np.maximum(high_d - np.roll(high_d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_d, 1) - low_d) > (high_d - np.roll(high_d, 1)), 
                        np.maximum(np.roll(low_d, 1) - low_d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(adx)] = 0
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    # Trend filter: ADX > 25
    trend_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for Camarilla and volume filter
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3, volume filter, trending market
            long_cond = (close[i] > camarilla_r3[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price breaks below Camarilla S3, volume filter, trending market
            short_cond = (close[i] < camarilla_s3[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla S3
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla R3
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals