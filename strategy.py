#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Weekly Donchian channels (20-week) capture major trend breakouts.
Daily timeframe provides entry precision with volume confirmation and ADX filter
to avoid false breakouts in ranging markets. Works in both bull and bear markets
by following breakout direction. Target: 10-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_highest(arr, period):
    """Calculate rolling highest with NaN for insufficient data"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.max(arr[i-period+1:i+1])
    return result

def calculate_lowest(arr, period):
    """Calculate rolling lowest with NaN for insufficient data"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.min(arr[i-period+1:i+1])
    return result

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
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
    dm_plus_smooth = np.zeros_like(high)
    dm_minus_smooth = np.zeros_like(high)
    
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < period:
            adx[i] = np.nan
        elif i == period:
            adx[i] = np.mean(dx[1:i+1])
        else:
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
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = calculate_highest(high_1w, 20)
    donchian_low = calculate_lowest(low_1w, 20)
    
    # Align to daily timeframe (use previous week's levels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    # ADX trend filter: only trade when ADX > 25 (trending market)
    adx = calculate_adx(high, low, close, 14)
    trending_market = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume in trending market
            if (close[i] > donchian_high_aligned[i] and 
                vol_spike[i] and 
                trending_market[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly Donchian low with volume in trending market
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_spike[i] and 
                  trending_market[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly Donchian high or volatility spike ends
            if close[i] < donchian_high_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly Donchian low or volatility spike ends
            if close[i] > donchian_low_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume_ADXFilter"
timeframe = "1d"
leverage = 1.0