#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian breakout with 12h ADX trend filter and volume confirmation.
Long when price breaks above Donchian(20) high and 12h ADX > 25 (trending).
Short when price breaks below Donchian(20) low and 12h ADX > 25.
Volume confirmation: current volume > 1.5x 20-period average.
Exits when price crosses opposite Donchian band or ADX < 20 (trend weak).
Designed for 15-25 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    tr0 = np.abs(high[1:] - low[1:])
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    tr = np.concatenate([np.array([np.nan]), tr])
    
    atr = np.full(len(high), np.nan)
    atr[period] = np.nanmean(tr[1:period+1])
    
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index."""
    if len(high) < period * 2:
        return np.full(len(high), np.nan)
    
    # Calculate True Range
    tr0 = np.abs(high[1:] - low[1:])
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    tr = np.concatenate([np.array([np.nan]), tr])
    
    # Calculate Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = np.concatenate([np.array([0]), plus_dm])
    minus_dm = np.concatenate([np.array([0]), minus_dm])
    
    # Smooth TR, +DM, -DM
    atr = np.full(len(high), np.nan)
    plus_dm_smooth = np.full(len(high), np.nan)
    minus_dm_smooth = np.full(len(high), np.nan)
    
    atr[period] = np.nanmean(tr[1:period+1])
    plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
    
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period - 1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period - 1) + minus_dm[i]) / period
    
    # Calculate Directional Indicators
    plus_di = np.full(len(high), np.nan)
    minus_di = np.full(len(high), np.nan)
    
    for i in range(period, len(high)):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.full(len(high), np.nan)
    adx = np.full(len(high), np.nan)
    
    for i in range(period, len(high)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx[2*period-1] = np.nanmean(dx[period:2*period])
    
    for i in range(2*period, len(high)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_donchian(high, low, period):
    """Calculate Donchian Channel."""
    upper = np.full(len(high), np.nan)
    lower = np.full(len(high), np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    adx_14_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align to 6h timeframe
    adx_14_12h_6h = align_htf_to_ltf(prices, df_12h, adx_14_12h)
    
    # Calculate Donchian(20) on 6h
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need ADX and Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_12h_6h[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper, ADX > 25, volume confirmation
            if close[i] > donchian_upper[i] and adx_14_12h_6h[i] > 25 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, ADX > 25, volume confirmation
            elif close[i] < donchian_lower[i] and adx_14_12h_6h[i] > 25 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian lower or ADX < 20 (trend weak)
            if close[i] < donchian_lower[i] or adx_14_12h_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian upper or ADX < 20 (trend weak)
            if close[i] > donchian_upper[i] or adx_14_12h_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hADX_Volume"
timeframe = "6h"
leverage = 1.0