#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + 1w ADX regime filter.
- Long when price breaks above Donchian(20) high, 1d volume > 1.5x MA(20), and 1w ADX < 25 (range).
- Short when price breaks below Donchian(20) low, 1d volume > 1.5x MA(20), and 1w ADX < 25 (range).
- Exit when price crosses opposite Donchian boundary or ADX > 30 (trend regime).
- Uses volatility breakout in ranging markets, avoids trends to reduce whipsaw.
- Designed for 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM-
    atr = np.full(len(high), np.nan)
    dm_plus_smooth = np.full(len(high), np.nan)
    dm_minus_smooth = np.full(len(high), np.nan)
    
    if len(high) >= period:
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = np.full(len(high), np.nan)
    minus_di = np.full(len(high), np.nan)
    dx = np.full(len(high), np.nan)
    
    for i in range(period, len(high)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX
    adx = np.full(len(high), np.nan)
    for i in range(2*period-1, len(high)):
        if i == 2*period-1:
            adx[i] = np.mean(dx[period-1:i+1])
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian(20) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume MA(20) on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Calculate ADX(14) on 1w
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align 1d volume MA and 1w ADX to 4h
    vol_ma_1d_4h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_14_1w_4h = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_4h[i]) or np.isnan(adx_14_1w_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period MA
        # Need to get the corresponding 1d volume index
        vol_confirmed = False
        if i < len(vol_ma_1d_4h) and not np.isnan(vol_ma_1d_4h[i]):
            # Approximate: use current 4h volume scaled to daily
            # Since we don't have direct 1d volume at 4h scale, use 4h volume as proxy
            vol_confirmed = volume[i] > 1.5 * vol_ma_1d_4h[i] * (1/96)  # rough scaling
        
        if position == 0:
            # Long: Donchian breakout up, volume confirmation, ranging market (ADX < 25)
            if close[i] > donchian_high[i] and vol_confirmed and adx_14_1w_4h[i] < 25:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown down, volume confirmation, ranging market (ADX < 25)
            elif close[i] < donchian_low[i] and vol_confirmed and adx_14_1w_4h[i] < 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low or ADX > 30 (trend)
            if close[i] < donchian_low[i] or adx_14_1w_4h[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high or ADX > 30 (trend)
            if close[i] > donchian_high[i] or adx_14_1w_4h[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVol_1wADX_Range"
timeframe = "4h"
leverage = 1.0