#!/usr/bin/env python3
"""
4h Williams %R(14) + Volume Spike + ADX(14) Trend Filter
- Williams %R < -80 for long, > -20 for short in trending markets (ADX > 25)
- Volume > 1.5x 20-period average for confirmation
- Exit when Williams %R crosses back through -50 (momentum fade)
- Designed for 20-50 trades/year with disciplined risk control
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R indicator."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    highest_high = np.full(len(high), np.nan)
    lowest_low = np.full(len(high), np.nan)
    
    for i in range(period-1, len(high)):
        highest_high[i] = np.max(high[i-period+1:i+1])
        lowest_low[i] = np.min(low[i-period+1:i+1])
    
    williams_r = np.full(len(high), np.nan)
    for i in range(period-1, len(high)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50  # when no range
    
    return williams_r

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index with Wilder's smoothing."""
    if len(high) < period * 2:
        return np.full(len(high), np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM using Wilder's smoothing
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    if len(dm_plus) >= period:
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        for i in range(period, len(dm_plus)):
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = np.full(len(dm_plus), np.nan)
    minus_di = np.full(len(dm_minus), np.nan)
    for i in range(period, len(atr)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
    
    # DX and ADX
    dx = np.full(len(plus_di), np.nan)
    for i in range(period, len(plus_di)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.full(len(dx), np.nan)
    if len(dx) >= 2 * period - 1:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_sma(arr, period):
    """Calculate Simple Moving Average with NaN for insufficient data."""
    sma = np.full(len(arr), np.nan)
    if len(arr) >= period:
        for i in range(period-1, len(arr)):
            sma[i] = np.mean(arr[i-period+1:i+1])
    return sma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams %R and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 4h
    williams_r_4h = calculate_williams_r(high_4h, low_4h, close_4h, 14)
    
    # Calculate 1-day ADX(14)
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 4h volume moving average (20-period)
    vol_ma_4h = calculate_sma(volume_4h, 20)
    
    # Align 1d ADX to 4h timeframe
    adx_14_1d_4h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Align 4h Williams %R and volume MA to 4h (no alignment needed)
    williams_r_4h_aligned = williams_r_4h
    vol_ma_4h_aligned = vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need sufficient data for Williams %R, ADX, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_4h_aligned[i]) or np.isnan(adx_14_1d_4h[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 4h volume for current 4h bar
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        vol_spike = vol_4h_aligned[i] > 1.5 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), volume spike, ADX > 25
            if williams_r_4h_aligned[i] < -80 and vol_spike and adx_14_1d_4h[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), volume spike, ADX > 25
            elif williams_r_4h_aligned[i] > -20 and vol_spike and adx_14_1d_4h[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses back above -50 (momentum fade)
            if williams_r_4h_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses back below -50 (momentum fade)
            if williams_r_4h_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR14_ADX14_VolumeSpike"
timeframe = "4h"
leverage = 1.0