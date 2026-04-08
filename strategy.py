#!/usr/bin/env python3
"""
6h_1w_1d_adx_alligator_v1
Hypothesis: 6-hour strategy combining weekly ADX trend strength with daily Alligator (SMAs) for trend-following entries.
Long when weekly ADX > 25 (strong trend) and price > daily Alligator Jaw (13-period SMA shifted 8 bars) with bullish alignment (Teeth > Lips).
Short when weekly ADX > 25 and price < daily Alligator Jaw with bearish alignment (Teeth < Lips).
Exit when ADX weakens (<20) or Alligator alignment reverses.
Uses weekly trend filter to avoid whipsaws in sideways markets and Alligator for precise entry timing.
Designed for low trade frequency (12-37/year) to minimize fee impact in 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_adx_alligator_v1"
timeframe = "6h"
leverage = 1.0

def calculate_sma(arr, period):
    """Calculate Simple Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    sma = np.full_like(arr, np.nan, dtype=float)
    for i in range(period-1, len(arr)):
        sma[i] = np.mean(arr[i-(period-1):i+1])
    return sma

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr = np.zeros_like(close)
    plus_di = np.zeros_like(close)
    minus_di = np.zeros_like(close)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    plus_dm_sum = np.sum(plus_dm[:period])
    minus_dm_sum = np.sum(minus_dm[:period])
    
    for i in range(period, len(close)):
        # Wilder smoothing
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
        
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_sum / atr[i]
            minus_di[i] = 100 * minus_dm_sum / atr[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # DX and ADX
    dx = np.zeros_like(close)
    for i in range(len(close)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    adx = np.full_like(close, np.nan, dtype=float)
    adx[2*period-2] = np.mean(dx[period-1:2*period-1])
    for i in range(2*period-1, len(close)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Alligator (SMAs)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly ADX for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Daily Alligator components (SMAs with specific shifts)
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMA shifted 8 bars
    jaw_1d = calculate_sma(close_1d, 13)
    jaw_1d = np.roll(jaw_1d, 8)  # Shift forward 8 bars
    jaw_1d[:8] = np.nan  # First 8 values invalid after shift
    # Teeth: 8-period SMA shifted 5 bars
    teeth_1d = calculate_sma(close_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)  # Shift forward 5 bars
    teeth_1d[:5] = np.nan
    # Lips: 5-period SMA shifted 3 bars
    lips_1d = calculate_sma(close_1d, 5)
    lips_1d = np.roll(lips_1d, 3)  # Shift forward 3 bars
    lips_1d[:3] = np.nan
    
    # Align Alligator to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(jaw_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        adx = adx_1w_aligned[i]
        price = close[i]
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        
        # Alligator alignment signals
        bullish_alignment = teeth > lips  # Teeth above Lips = bullish
        bearish_alignment = teeth < lips  # Teeth below Lips = bearish
        
        if position == 1:  # Long
            # Exit: ADX weakens (<20) or Alligator alignment turns bearish
            if adx < 20.0 or not bullish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: ADX weakens (<20) or Alligator alignment turns bullish
            if adx < 20.0 or not bearish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Strong trend (ADX > 25) + price above Jaw + bullish alignment
            if adx > 25.0 and price > jaw and bullish_alignment:
                position = 1
                signals[i] = 0.25
            # Enter short: Strong trend (ADX > 25) + price below Jaw + bearish alignment
            elif adx > 25.0 and price < jaw and bearish_alignment:
                position = -1
                signals[i] = -0.25
    
    return signals