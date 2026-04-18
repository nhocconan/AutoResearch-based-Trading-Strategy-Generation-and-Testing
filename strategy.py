#!/usr/bin/env python3
"""
4h Williams Alligator with Volume Spike and ADX Trend Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction. 
When Lips cross above Teeth and Jaw in low volatility, it signals trend start. 
Volume spike confirms participation. ADX > 25 ensures trending market.
Works in both bull and bear by following Alligator alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_wma(data, period):
    """Weighted Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan)
    weights = np.arange(1, period + 1)
    wma = np.convolve(data, weights[::-1], mode='valid') / weights.sum()
    result = np.full_like(data, np.nan)
    result[period-1:] = wma
    return result

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    dm_plus_smooth = np.zeros_like(high)
    dm_minus_smooth = np.zeros_like(high)
    dm_plus_smooth[0] = dm_plus[0]
    dm_minus_smooth[0] = dm_minus[0]
    for i in range(1, len(high)):
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(high)
    dx_sum = plus_di + minus_di
    dx = np.where(dx_sum != 0, 100 * np.abs(plus_di - minus_di) / dx_sum, 0)
    
    adx = np.zeros_like(high)
    if len(dx) >= period:
        adx[period-1] = np.mean(dx[:period])
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Williams Alligator on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d) / 2  # Typical price for Alligator
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMMA
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_1d, 13)
    teeth = smma(median_1d, 8)
    lips = smma(median_1d, 5)
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # ADX trend filter on 4h
    adx = calculate_adx(high, low, close, 14)
    trending = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume spike + trend
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                vol_spike[i] and 
                trending[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume spike + trend
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  vol_spike[i] and 
                  trending[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips < Teeth or Teeth < Jaw)
            if lips_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips > Teeth or Teeth > Jaw)
            if lips_aligned[i] > teeth_aligned[i] or teeth_aligned[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0