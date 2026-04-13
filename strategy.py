#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Camarilla pivot breakout with volume confirmation and session filter.
# Uses Camarilla pivot levels (S3/S4 for shorts, R3/R4 for longs) from 4h and 1d timeframes.
# Requires volume > 1.5x average and trading session 08-20 UTC.
# Takes long when price breaks above R4 with volume, short when breaks below S4.
# Exit when price returns to pivot point or opposite Camarilla level.
# Designed for low frequency: ~20-50 trades per year on 1h timeframe.
# Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    range_4h = high_4h - low_4h
    
    # Camarilla: R4 = C + (H-L)*1.5, R3 = C + (H-L)*1.25, S3 = C - (H-L)*1.25, S4 = C - (H-L)*1.5
    r4_4h = close_4h + range_4h * 1.5
    r3_4h = close_4h + range_4h * 1.25
    s3_4h = close_4h - range_4h * 1.25
    s4_4h = close_4h - range_4h * 1.5
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    
    # Calculate Camarilla levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    r4_1d = close_1d + range_1d * 1.5
    r3_1d = close_1d + range_1d * 1.25
    s3_1d = close_1d - range_1d * 1.25
    s4_1d = close_1d - range_1d * 1.5
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 1h
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate average volume (24-period = 24 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(r4_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or 
            np.isnan(s3_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(pivot_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Price breaks above R4 of both 4h and 1d with volume
            if (price > r4_4h_aligned[i] and price > r4_1d_aligned[i] and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S4 of both 4h and 1d with volume
            elif (price < s4_4h_aligned[i] and price < s4_1d_aligned[i] and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to pivot or breaks below S3
            if (price <= pivot_4h_aligned[i] or price <= pivot_1d_aligned[i] or 
                price < s3_4h_aligned[i] or price < s3_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price returns to pivot or breaks above R3
            if (price >= pivot_4h_aligned[i] or price >= pivot_1d_aligned[i] or 
                price > r3_4h_aligned[i] or price > r3_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Camarilla_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0