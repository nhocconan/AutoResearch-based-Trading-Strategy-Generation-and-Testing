#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirmation ensures institutional participation. 
ADX > 25 filters for trending markets, avoiding whipsaws in ranging conditions. Works in both bull and bear markets 
by following breakout direction regardless of trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period:
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
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / (atr[i] + 1e-10) * 100
        minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / (atr[i] + 1e-10) * 100
    
    # DX and ADX
    dx = np.zeros_like(high)
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    
    adx = np.zeros_like(high)
    adx[period-1] = np.mean(dx[:period]) if period <= len(dx) else 0
    for i in range(period, len(dx)):
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
    
    # Get daily data for volume average (more stable)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels
    upper, lower = calculate_donchian_channels(high, low, 20)
    
    # Calculate ADX for trend strength
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper band with volume and trend
            if (close[i] > upper[i] and 
                vol_spike[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below lower band with volume and trend
            elif (close[i] < lower[i] and 
                  vol_spike[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below middle or trend weakens
            mid = (upper[i] + lower[i]) / 2
            if close[i] < mid or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above middle or trend weakens
            mid = (upper[i] + lower[i]) / 2
            if close[i] > mid or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0