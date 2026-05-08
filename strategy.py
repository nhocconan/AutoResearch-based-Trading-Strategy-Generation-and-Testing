#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot S3/R3 breakout with 12h volume spike and ADX trend filter.
# Long when price breaks above R3 AND 12h volume > 1.5x 24-period average AND ADX(14) > 25.
# Short when price breaks below S3 AND 12h volume > 1.5x 24-period average AND ADX(14) > 25.
# Exit when price crosses back inside the S3-R3 range.
# Uses 4h primary timeframe as specified, with 12h volume and ADX for higher timeframe context.
# Target: 100-200 total trades over 4 years (25-50/year) with controlled frequency to avoid fee drag.

name = "4h_Camarilla_R3S3_12hVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for volume and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily data
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: R3 = close + (high-low)*1.1/2, S3 = close - (high-low)*1.1/2
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    camarilla_range = (high_d - low_d) * 1.1
    r3_d = close_d + camarilla_range / 2
    s3_d = close_d - camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_d, r3_d)
    s3_4h = align_htf_to_ltf(prices, df_d, s3_d)
    
    # 12h volume filter: current volume > 1.5x 24-period average
    volume_12h = df_12h['volume'].values
    vol_ma24_12h = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma24_12h)
    volume_filter = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    # 12h ADX(14) for trend strength
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_12h[0] - low_12h[0]
    
    # Directional Movement
    plus_dm = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_12h
    minus_di = 100 * minus_dm_smooth / atr_12h
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_12h[np.isnan(adx_12h)] = 0
    
    # Align ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Trend filter: ADX > 25
    trend_filter = adx_4h > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 14)  # Sufficient warmup for volume MA and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, volume filter, trending market
            long_cond = (close[i] > r3_4h[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price breaks below S3, volume filter, trending market
            short_cond = (close[i] < s3_4h[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below S3
            if close[i] < s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above R3
            if close[i] > r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals