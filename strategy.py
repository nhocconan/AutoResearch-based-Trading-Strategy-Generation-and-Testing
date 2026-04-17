#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d Volume Spike + ADX Filter
Long: Price breaks above Donchian(20) high + volume > 1.5x 1d avg volume + ADX > 20
Short: Price breaks below Donchian(20) low + volume > 1.5x 1d avg volume + ADX > 20
Exit: Opposite Donchian breakout or ADX < 15
Uses Donchian channels for trend-following breakouts with volume confirmation and trend strength filter.
Designed to capture strong momentum moves in both bull and bear markets while avoiding choppy periods.
Target: 20-50 total trades over 4 years (5-12.5/year)
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
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period SMA)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate ADX(14) for trend strength filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])
    plus_dm_smooth[13] = np.sum(plus_dm[1:14])
    minus_dm_smooth[13] = np.sum(minus_dm[1:14])
    
    # Wilder's smoothing
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX smoothing
    adx = np.zeros(n)
    adx[27] = np.mean(dx[28:29]) if len(dx) > 28 else 0  # First ADX value
    for i in range(28, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Donchian Channel (20-period)
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 28)  # Need ADX and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(avg_volume_1d_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_1d_aligned[i]
        adx_val = adx[i]
        upper = donch_high[i]
        lower = donch_low[i]
        
        if position == 0:
            # Long: Break above Donchian high + volume spike + strong trend
            if price > upper and vol > 1.5 * avg_vol and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + volume spike + strong trend
            elif price < lower and vol > 1.5 * avg_vol and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Break below Donchian low or weak trend
            if price < lower or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Break above Donchian high or weak trend
            if price > upper or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dVolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0