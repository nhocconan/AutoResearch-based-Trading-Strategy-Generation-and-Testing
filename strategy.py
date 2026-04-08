#!/usr/bin/env python3
# [24879] 6h_12h1d_alligator_v1
# Hypothesis: 6-hour strategy using Williams Alligator (13,8,5 SMAs) on 12h and 1d with 60-period ADX trend filter.
# Long when price > Alligator Jaw (13-period SMA) on 12h AND ADX > 25 on 1d.
# Short when price < Alligator Jaw (13-period SMA) on 12h AND ADX > 25 on 1d.
# Exit when price crosses Alligator Teeth (8-period SMA) on 12h OR ADX drops below 20 on 1d.
# Uses Alligator to identify trend direction and ADX to filter strong trends, avoiding whipsaws in ranging markets.
# Works in both bull (trend following) and bear (trend following short) markets.
# Target: 15-35 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h1d_alligator_v1"
timeframe = "6h"
leverage = 1.0

def calculate_sma(arr, period):
    """Calculate SMA with proper handling"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    
    sma = np.full_like(arr, np.nan, dtype=float)
    for i in range(period-1, len(arr)):
        sma[i] = np.mean(arr[i-period+1:i+1])
    return sma

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
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
    
    # Initial values (simple average)
    if len(close) >= period:
        atr[period-1] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = np.full_like(close, np.nan, dtype=float)
    minus_di = np.full_like(close, np.nan, dtype=float)
    dx = np.full_like(close, np.nan, dtype=float)
    
    for i in range(period, len(close)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX (smoothed DX)
    adx = np.full_like(close, np.nan, dtype=float)
    if len(close) >= 2*period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12-hour and 1-day data for context
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Alligator (13,8,5 SMAs on median price)
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    jaw_12h = calculate_sma(median_price_12h, 13)   # Jaw (13-period SMA)
    teeth_12h = calculate_sma(median_price_12h, 8)  # Teeth (8-period SMA)
    lips_12h = calculate_sma(median_price_12h, 5)   # Lips (5-period SMA)
    
    # Calculate 1d ADX
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align indicators to 6-hour timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        jaw = jaw_12h_aligned[i]
        teeth = teeth_12h_aligned[i]
        adx = adx_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below Alligator Teeth OR ADX drops below 20 (trend weakening)
            if price < teeth or adx < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above Alligator Teeth OR ADX drops below 20 (trend weakening)
            if price > teeth or adx < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above Alligator Jaw AND strong trend (ADX > 25)
            if price > jaw and adx > 25:
                position = 1
                signals[i] = 0.25
            # Enter short: price below Alligator Jaw AND strong trend (ADX > 25)
            elif price < jaw and adx > 25:
                position = -1
                signals[i] = -0.25
    
    return signals