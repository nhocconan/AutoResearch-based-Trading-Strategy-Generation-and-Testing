#!/usr/bin/env python3
"""
4h Camarilla Pivot (R1/S1) Breakout with Volume Spike and ADX Filter
- Uses 1-day Camarilla pivot levels (R1, S1) for breakout signals
- Confirms with 4-hour volume > 1.8x 20-period average
- Filters by 1-day ADX > 20 to ensure trending market
- Exits on opposite Camarilla level touch
- Designed for 20-40 trades/year (80-160 total) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    multiplier = 1.1 / 12
    c = close
    h = high
    l = low
    r4 = c + range_val * 1.1 / 2
    r3 = c + range_val * 1.1 / 4
    r2 = c + range_val * 1.1 / 6
    r1 = c + range_val * multiplier
    s1 = c - range_val * multiplier
    s2 = c - range_val * 1.1 / 6
    s3 = c - range_val * 1.1 / 4
    s4 = c - range_val * 1.1 / 2
    return r1, r2, r3, r4, s1, s2, s3, s4

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
    
    # Get 1d data for Camarilla and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Camarilla levels (using previous day's OHLC)
    r1_1d = np.full(len(high_1d), np.nan)
    r2_1d = np.full(len(high_1d), np.nan)
    s1_1d = np.full(len(high_1d), np.nan)
    s2_1d = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        r1, r2, _, _, s1, s2, _, _ = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        r1_1d[i] = r1
        r2_1d[i] = r2
        s1_1d[i] = s1
        s2_1d[i] = s2
    
    # Calculate 1-day ADX (14-period)
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Get 4h data for volume
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h volume moving average (20-period)
    vol_ma_4h = calculate_sma(volume_4h, 20)
    
    # Align 1d indicators to 4h timeframe
    r1_1d_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_4h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s1_1d_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_4h = align_htf_to_ltf(prices, df_1d, s2_1d)
    adx_14_1d_4h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Align 4h volume and volume MA to 4h timeframe
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_4h[i]) or np.isnan(s1_1d_4h[i]) or 
            np.isnan(volume_4h_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(adx_14_1d_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.8x 20-period average
        vol_spike = volume_4h_aligned[i] > 1.8 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Long: price above R1, volume spike, ADX > 20
            if close[i] > r1_1d_4h[i] and vol_spike and adx_14_1d_4h[i] > 20:
                signals[i] = 0.25
                position = 1
            # Short: price below S1, volume spike, ADX > 20
            elif close[i] < s1_1d_4h[i] and vol_spike and adx_14_1d_4h[i] > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches or goes below S1
            if close[i] < s1_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or goes above R1
            if close[i] > r1_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSpike_ADX20"
timeframe = "4h"
leverage = 1.0