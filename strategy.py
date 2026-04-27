#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d ADX Trend Filter and Volume Confirmation
Hypothesis: Donchian(20) breakouts on 12h capture medium-term trends. Filtering by 1d ADX>25 ensures we only trade in trending markets (works in both bull and bear). Volume confirmation (>1.5x average) avoids false breakouts. Target: 15-25 trades/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for 1d
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index 0
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = np.full_like(tr, np.nan)
        plus_dm_smooth = np.full_like(plus_dm, np.nan)
        minus_dm_smooth = np.full_like(minus_dm, np.nan)
        
        # Initial values (simple average)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])  # skip first NaN
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
            
            # Wilder smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        dx = np.full_like(atr, np.nan)
        valid = (atr != 0) & ~np.isnan(atr)
        dx[valid] = (np.abs(plus_dm_smooth[valid] - minus_dm_smooth[valid]) / atr[valid]) * 100
        
        # ADX: smoothed DX
        adx = np.full_like(dx, np.nan)
        if len(dx) >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])  # initial ADX
            for i in range(2*period-1, len(dx)):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels (20-period) on 12h
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume MA (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), ADX (14+14=28 bars), volume MA (20)
    start_idx = max(19, 27, 19)  # 27 for ADX (14+13 for DX smoothing)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian upper channel with volume and trend
            if price > highest_high[i] and vol_filter and trend_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower channel with volume and trend
            elif price < lowest_low[i] and vol_filter and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian lower channel (stop and reverse)
            if price < lowest_low[i]:
                signals[i] = -size  # reverse to short
                position = -1
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian upper channel (stop and reverse)
            if price > highest_high[i]:
                signals[i] = size  # reverse to long
                position = 1
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_1dADX25_VolumeFilter"
timeframe = "12h"
leverage = 1.0