#!/usr/bin/env python3
"""
4h_1d_camarilla_pivot_v8
Hypothesis: 4-hour strategy using 1-day Camarilla pivot with volume confirmation and ADX trend filter.
Long when price breaks above 1-day R3 with volume > 1.8x average and ADX > 25.
Short when price breaks below 1-day S3 with volume > 1.8x average and ADX > 25.
Exit when price crosses opposite 1-day level (S3 for long, R3 for short) or volume drops below 1.2x average.
Uses daily pivot levels for structural support/resistance in both bull and bear markets.
Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_v8"
timeframe = "4h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX with proper handling"""
    if len(high) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = np.full_like(close, np.nan, dtype=float)
    dm_plus_smooth = np.full_like(close, np.nan, dtype=float)
    dm_minus_smooth = np.full_like(close, np.nan, dtype=float)
    
    # Initial average
    if len(close) >= period:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder smoothing
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    di_plus = np.full_like(close, np.nan, dtype=float)
    di_minus = np.full_like(close, np.nan, dtype=float)
    dx = np.full_like(close, np.nan, dtype=float)
    
    for i in range(period, len(close)):
        if atr[i] > 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if di_plus[i] + di_minus[i] > 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX
    adx = np.full_like(close, np.nan, dtype=float)
    for i in range(2 * period, len(close)):
        if not np.isnan(dx[i]) and i >= 2 * period:
            if i == 2 * period:
                adx[i] = np.nanmean(dx[period:i+1])
            else:
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Pivot (using previous 1-day bar's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # 1-day support/resistance levels (Camarilla S3/R3)
    S3_1d = pivot_1d - (range_1d * 1.1 / 4)  # 1d S3
    R3_1d = pivot_1d + (range_1d * 1.1 / 4)  # 1d R3
    
    # Calculate ADX for trend filter
    adx = calculate_adx(high, low, close, 14)
    
    # Align indicators to 4-hour timeframe
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S3_1d_aligned[i]) or np.isnan(R3_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S3 = S3_1d_aligned[i]
        R3 = R3_1d_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 1-day S3 or volume drops below 1.2x average
            if price < S3 or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 1-day R3 or volume drops below 1.2x average
            if price > R3 or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 1-day R3 with volume expansion and strong trend
            if price > R3 and vol_ratio > 1.8 and adx_val > 25:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 1-day S3 with volume expansion and strong trend
            elif price < S3 and vol_ratio > 1.8 and adx_val > 25:
                position = -1
                signals[i] = -0.25
    
    return signals