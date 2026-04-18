#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with Volume Spike and 1d ADX Filter
- Uses 12-hour Camarilla pivot levels (R1, S1) from prior 1-day calculation
- Confirms with 12-hour volume > 2.0x 20-period average
- Filters by 1-day ADX > 25 to ensure trending market
- Exits on opposite Camarilla level touch
- Designed for 12-37 trades/year (50-150 total) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC."""
    # Pivot point
    pivot = (high + low + close) / 3.0
    # Range
    range_hl = high - low
    # Camarilla levels
    r1 = close + range_hl * 1.1 / 12
    s1 = close - range_hl * 1.1 / 12
    r2 = close + range_hl * 1.1 / 6
    s2 = close - range_hl * 1.1 / 6
    r3 = close + range_hl * 1.1 / 4
    s3 = close - range_hl * 1.1 / 4
    r4 = close + range_hl * 1.1 / 2
    s4 = close - range_hl * 1.1 / 2
    return r1, s1, r2, s2, r3, s3, r4, s4

def calculate_sma(arr, period):
    """Calculate Simple Moving Average with NaN for insufficient data."""
    sma = np.full(len(arr), np.nan)
    if len(arr) >= period:
        for i in range(period-1, len(arr)):
            sma[i] = np.mean(arr[i-period+1:i+1])
    return sma

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index with proper smoothing."""
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
    
    # Smooth TR and DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Camarilla pivot levels (R1, S1)
    r1_1d, s1_1d, r2_1d, s2_1d, r3_1d, s3_1d, r4_1d, s4_1d = calculate_camarilla_pivot(high_1d, low_1d, close_1d)
    
    # Calculate 1-day ADX (14-period)
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Get 12h data for price and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_12h = calculate_sma(volume_12h, 20)
    
    # Align 1d indicators to 12h timeframe
    r1_1d_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    adx_14_1d_12h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Align 12h volume MA to 12h timeframe (no alignment needed)
    vol_ma_12h_aligned = vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need sufficient data for volume MA and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_12h[i]) or np.isnan(s1_1d_12h[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(adx_14_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 12h volume for current 12h bar
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        
        # Volume confirmation: current 12h volume > 2.0x 20-period average
        vol_spike = vol_12h_aligned[i] > 2.0 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Long: price above R1 level, volume spike, ADX > 25
            if close[i] > r1_1d_12h[i] and vol_spike and adx_14_1d_12h[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price below S1 level, volume spike, ADX > 25
            elif close[i] < s1_1d_12h[i] and vol_spike and adx_14_1d_12h[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches or goes below S1 level
            if close[i] < s1_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or goes above R1 level
            if close[i] > r1_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_VolumeSpike_ADX14"
timeframe = "12h"
leverage = 1.0