#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume confirmation and ADX trend filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 1.3x 20-period average AND ADX(14) > 25.
# Short when price breaks below Camarilla S3 AND 1d volume > 1.3x 20-period average AND ADX(14) > 25.
# Exit when price crosses back inside the Camarilla (H4-L4) range.
# Uses 4h timeframe as specified, with 1d volume and ADX for higher timeframe context.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled frequency to avoid fee drag.

name = "4h_Camarilla_R3S3_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for volume and ADX
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar using prior 1d OHLC
    # We'll compute daily Camarilla levels first, then align to 4h
    # Camarilla levels based on prior day's OHLC
    open_d = df_d['open'].values
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Typical price for Camarilla calculation
    # Using (high + low + close) / 3 as pivot
    typical_price = (high_d + low_d + close_d) / 3.0
    range_d = high_d - low_d
    
    # Camarilla levels
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    r3_d = close_d + range_d * 1.1 / 4.0
    s3_d = close_d - range_d * 1.1 / 4.0
    h4_d = close_d + range_d * 1.1 / 6.0  # H4 level
    l4_d = close_d - range_d * 1.1 / 6.0  # L4 level
    
    # Align Camarilla levels to 4h timeframe (using prior day's close)
    r3 = align_htf_to_ltf(prices, df_d, r3_d)
    s3 = align_htf_to_ltf(prices, df_d, s3_d)
    h4 = align_htf_to_ltf(prices, df_d, h4_d)
    l4 = align_htf_to_ltf(prices, df_d, l4_d)
    
    # Daily volume filter: current volume > 1.3x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.3 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Daily ADX(14) for trend strength
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
    
    start_idx = 20  # Need at least 20 days for volume MA and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3, volume filter, trending market
            long_cond = (close[i] > r3[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price breaks below Camarilla S3, volume filter, trending market
            short_cond = (close[i] < s3[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below H4 level (or S3 for tighter stop)
            if close[i] < h4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above L4 level (or R3 for tighter stop)
            if close[i] > l4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals