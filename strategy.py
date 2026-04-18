#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and 1d ADX Filter
Hypothesis: Donchian channel breakouts on 12h timeframe capture medium-term momentum moves.
Volume confirms institutional participation. 1d ADX filter ensures we only trade in trending markets,
avoiding whipsaws in ranging conditions. Works in both bull and bear markets by following breakout direction.
Target: 12-37 trades per year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    
    plus_dm_smooth = np.zeros_like(high)
    minus_dm_smooth = np.zeros_like(high)
    
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    
    for i in range(1, len(high)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(high)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    adx = np.zeros_like(high)
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channel (20-period)
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(len(high)):
        if i < 19:
            highest_high[i] = np.max(high[0:i+1]) if i >= 0 else high[i]
            lowest_low[i] = np.min(low[0:i+1]) if i >= 0 else low[i]
        else:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 19:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 25 (trending market)
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long breakout: price breaks above 20-period high with volume in trending market
            if (close[i] > highest_high[i] and vol_spike[i] and trending):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 20-period low with volume in trending market
            elif (close[i] < lowest_low[i] and vol_spike[i] and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 20-period low or volatility ends
            if close[i] < lowest_low[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 20-period high or volatility ends
            if close[i] > highest_high[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0