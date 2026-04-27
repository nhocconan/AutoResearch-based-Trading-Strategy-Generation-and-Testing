#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and ADX Trend Filter.
Long when: 1) Price breaks above Donchian upper band (20-period high), 2) ADX > 25 (trending), 3) Volume > 1.5x 20-period average.
Short when: 1) Price breaks below Donchian lower band (20-period low), 2) ADX > 25 (trending), 3) Volume > 1.5x 20-period average.
Exit when price returns to the opposite Donchian band or ADX < 20 (trend weakening).
Designed for 4h timeframe: targets 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+
    atr = np.full(n, np.nan)
    dm_plus_smooth = np.full(n, np.nan)
    dm_minus_smooth = np.full(n, np.nan)
    
    # Initial values (first 14 periods)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:15])
        dm_plus_smooth[13] = np.nanmean(dm_plus[1:15])
        dm_minus_smooth[13] = np.nanmean(dm_minus[1:15])
    
    # Wilder smoothing
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    for i in range(14, n):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
    
    # DX and ADX
    dx = np.full(n, np.nan)
    for i in range(14, n):
        if di_plus[i] + di_minus[i] != 0:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    adx = np.full(n, np.nan)
    if len(dx) >= 27:
        adx[26] = np.nanmean(dx[14:28])
        for i in range(27, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), ADX (27+14=41), volume MA (20)
    start_idx = 41
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        upper_band = donch_high[i]
        lower_band = donch_low[i]
        adx_val = adx[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        # Trend filter: ADX > 25
        trend_filter = adx_val > 25
        # Weak trend filter: ADX < 20
        weak_trend = adx_val < 20
        
        if position == 0:
            # Long: price breaks above upper band + trending + volume spike
            if price > upper_band and trend_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band + trending + volume spike
            elif price < lower_band and trend_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower band or trend weakens
            if price < lower_band or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to upper band or trend weakens
            if price > upper_band or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX_Trend"
timeframe = "4h"
leverage = 1.0