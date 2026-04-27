#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume and ADX Filter
Long when price breaks above Donchian(20) high with volume > 1.5x average and ADX > 25
Short when price breaks below Donchian(20) low with volume > 1.5x average and ADX > 25
Exit when price crosses the opposite Donchian boundary or ADX drops below 20
Designed for 12h timeframe to capture medium-term trends with volume confirmation
"""

import numpy as np
import pandas as pd
from mptf_data import get_htf_data, align_htf_to_ltf

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
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    adx_period = 14
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM-
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    if len(tr) >= adx_period:
        atr[adx_period-1] = np.mean(tr[:adx_period])
        dm_plus_smooth[adx_period-1] = np.mean(dm_plus[:adx_period])
        dm_minus_smooth[adx_period-1] = np.mean(dm_minus[:adx_period])
        
        for i in range(adx_period, len(tr)):
            atr[i] = (atr[i-1] * (adx_period-1) + tr[i]) / adx_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (adx_period-1) + dm_plus[i]) / adx_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (adx_period-1) + dm_minus[i]) / adx_period
    
    # Calculate DI+ and DI-
    di_plus = np.zeros_like(atr)
    di_minus = np.zeros_like(atr)
    dx = np.zeros_like(atr)
    
    mask = atr != 0
    di_plus[mask] = 100 * dm_plus_smooth[mask] / atr[mask]
    di_minus[mask] = 100 * dm_minus_smooth[mask] / atr[mask]
    
    di_sum = di_plus + di_minus
    mask = di_sum != 0
    dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / di_sum[mask]
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros_like(dx)
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.mean(dx[:adx_period])
        for i in range(adx_period, len(dx)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period) on 12h data
    donchian_period = 20
    dc_high = np.full(n, np.nan)
    dc_low = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        dc_high[i] = np.max(high[i - donchian_period + 1:i + 1])
        dc_low[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, ADX, and volume MA20
    start_idx = max(donchian_period, adx_period + adx_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # ADX filter: trending market
        adx_filter = adx_1d_aligned[i] > 20
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and ADX filter
            if price > dc_high[i] and vol_filter and adx_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and ADX filter
            elif price < dc_low[i] and vol_filter and adx_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian low OR ADX drops below 20
            if price < dc_low[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian high OR ADX drops below 20
            if price > dc_high[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_ADX20_Volume"
timeframe = "12h"
leverage = 1.0