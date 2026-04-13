#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume confirmation + ADX trend filter.
# Camarilla pivots provide precise S/R levels for mean-reversion in ranging markets.
# Long: price > S3 level + ADX < 25 (ranging) + volume > 1.5x avg volume
# Short: price < R3 level + ADX < 25 (ranging) + volume > 1.5x avg volume
# Exit: price crosses back through pivot levels or ADX > 25 (trending)
# Works in both bull and bear by using ADX to filter for ranging markets only.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 days for ADX(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # S2 = C - (Range * 1.1 / 6)
    # S3 = C - (Range * 1.1 / 4)
    # R1 = C + (Range * 1.1 / 12)
    # R2 = C + (Range * 1.1 / 6)
    # R3 = C + (Range * 1.1 / 4)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Calculate ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        # Initial values (simple average)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
            
            # Wilder's smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.full_like(tr, np.nan)
        di_minus = np.full_like(tr, np.nan)
        dx = np.full_like(tr, np.nan)
        
        valid = ~np.isnan(atr) & (atr != 0)
        di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        dx[valid] = 100 * np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])
        
        # ADX = smoothed DX
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Average volume (10-period = 5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(10, n):
        avg_volume[i] = np.mean(volume[i-10:i])
    
    # Align 1d indicators to 12h timeframe
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # Start after enough data for indicators
        # Skip if any required data is not ready
        if (np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        s3 = s3_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        adx = adx_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Ranging market filter: ADX < 25
        ranging = adx < 25
        
        if position == 0:
            # Long: price > S3 + ranging + volume confirmation
            if (price > s3 and 
                ranging and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price < R3 + ranging + volume confirmation
            elif (price < r3 and 
                  ranging and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < S3 or ADX > 25 (trending)
            if (price < s3 or
                adx > 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > R3 or ADX > 25 (trending)
            if (price > r3 or
                adx > 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Range_Volume"
timeframe = "12h"
leverage = 1.0