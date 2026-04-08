#!/usr/bin/env python3
"""
6h_1w_1d_adx_alligator_v1
Hypothesis: 6-hour strategy combining weekly ADX trend strength with daily Alligator for entry timing.
Long when weekly ADX > 25 (trending) + price above Alligator teeth (green).
Short when weekly ADX > 25 + price below Alligator teeth (red).
Uses weekly trend filter to avoid whipsaws and daily Alligator for precise entries.
Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_adx_alligator_v1"
timeframe = "6h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX with proper smoothing"""
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
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Initial values
    atr[period] = np.nanmean(tr[1:period+1])
    dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
    dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
    
    # Wilder smoothing
    for i in range(period + 1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # DI values
    di_plus = np.full_like(close, np.nan)
    di_minus = np.full_like(close, np.nan)
    dx = np.full_like(close, np.nan)
    
    valid = atr > 0
    di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
    
    dx_valid = (di_plus + di_minus) > 0
    dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
    
    # ADX (smoothed DX)
    adx = np.full_like(close, np.nan)
    if len(dx) >= 2 * period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Alligator (Smoothed Medians)"""
    if len(close) < jaw_period:
        return np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan)
    
    # Median price
    median = (high + low) / 2.0
    
    # Smoothed medians (SMMA)
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period - 1) + data[i]) / period
        return result
    
    jaw = smma(median, jaw_period)
    teeth = smma(median, teeth_period)
    lips = smma(median, lips_period)
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly and daily data for context
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly ADX for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Calculate daily Alligator for entry timing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    jaw_1d, teeth_1d, lips_1d = calculate_alligator(high_1d, low_1d, close_1d)
    
    # Align indicators to 6-hour timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        adx = adx_1w_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        price = close[i]
        
        # Determine Alligator state: green (lips > teeth) = bullish, red (lips < teeth) = bearish
        is_green = lips > teeth
        is_red = lips < teeth
        
        if position == 1:  # Long
            # Exit: trend weakens (ADX < 20) or Alligator turns red
            if adx < 20 or is_red:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: trend weakens (ADX < 20) or Alligator turns green
            if adx < 20 or is_green:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: strong trend (ADX > 25) + Alligator green
            if adx > 25 and is_green:
                position = 1
                signals[i] = 0.25
            # Enter short: strong trend (ADX > 25) + Alligator red
            elif adx > 25 and is_red:
                position = -1
                signals[i] = -0.25
    
    return signals