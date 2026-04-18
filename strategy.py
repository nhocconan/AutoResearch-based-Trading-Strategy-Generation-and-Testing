#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
Volume confirmation ensures institutional participation, while ADX filter 
avoids choppy markets. Works in both bull and bear markets by following 
breakout direction. Uses discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_highest(arr, period):
    """Calculate rolling highest with proper NaN handling"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.max(arr[i-period+1:i+1])
    return result

def calculate_lowest(arr, period):
    """Calculate rolling lowest with proper NaN handling"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.min(arr[i-period+1:i+1])
    return result

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_dm_smooth = np.zeros_like(high)
    minus_dm_smooth = np.zeros_like(high)
    for i in range(1, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Calculate Directional Indicators
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(high)
    for i in range(len(dx)):
        if i < period:
            adx[i] = np.nan
        elif i == period:
            adx[i] = np.mean(dx[1:period+1])
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
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe for trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channel (20-period) on 4h
    donchian_high = calculate_highest(high, 20)
    donchian_low = calculate_lowest(low, 20)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX indicates trending market (>25)
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume
            if (close[i] > donchian_high[i] and 
                vol_spike[i] and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below Donchian low with volume
            elif (close[i] < donchian_low[i] and 
                  vol_spike[i] and 
                  trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian low or volatility ends
            if close[i] < donchian_low[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian high or volatility ends
            if close[i] > donchian_high[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0