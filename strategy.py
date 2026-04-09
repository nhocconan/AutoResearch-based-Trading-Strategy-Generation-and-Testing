#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ADX(14) > 25 trend filter
# Uses 12h timeframe to reduce trade frequency and avoid fee drag. 
# Donchian breakouts capture breakouts in both bull and bear markets.
# Volume > 2x 20-period average confirms institutional participation.
# ADX > 25 filters for trending conditions to avoid false breakouts in ranging markets.
# Fixed position size 0.25 to limit drawdown and control risk.
# Designed for ~15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

name = "12h_donchian_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # True Range and Directional Movement
    tr = np.full(len(d_high), np.nan)
    dm_plus = np.full(len(d_high), np.nan)
    dm_minus = np.full(len(d_high), np.nan)
    
    for i in range(1, len(d_high)):
        # True Range
        tr0 = d_high[i] - d_low[i]
        tr1 = abs(d_high[i] - d_close[i-1])
        tr2 = abs(d_low[i] - d_close[i-1])
        tr[i] = max(tr0, tr1, tr2)
        
        # Directional Movement
        up_move = d_high[i] - d_high[i-1]
        down_move = d_low[i-1] - d_low[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0.0
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0.0
    
    # Smoothed averages with Wilder's smoothing
    tr14 = np.full(len(d_high), np.nan)
    dm_plus_14 = np.full(len(d_high), np.nan)
    dm_minus_14 = np.full(len(d_high), np.nan)
    
    if len(d_high) >= 14:
        tr14[13] = np.nansum(tr[1:14])
        dm_plus_14[13] = np.nansum(dm_plus[1:14])
        dm_minus_14[13] = np.nansum(dm_minus[1:14])
        
        for i in range(14, len(d_high)):
            tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI and DX
    di_plus = np.full(len(d_high), np.nan)
    di_minus = np.full(len(d_high), np.nan)
    dx = np.full(len(d_high), np.nan)
    
    for i in range(14, len(d_high)):
        if tr14[i] > 0:
            di_plus[i] = 100 * dm_plus_14[i] / tr14[i]
            di_minus[i] = 100 * dm_minus_14[i] / tr14[i]
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (smoothed DX)
    adx_1d = np.full(len(d_high), np.nan)
    if len(d_high) >= 28:
        adx_1d[27] = np.nansum(dx[14:28]) / 14
        for i in range(28, len(d_high)):
            adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Align daily ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(adx_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR ADX drops below 20 (trend weakening)
            if close[i] <= donchian_low[i] or adx_12h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR ADX drops below 20
            if close[i] >= donchian_high[i] or adx_12h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Donchian high with volume confirmation AND ADX > 25
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > donchian_high[i] and 
                vol_ratio > 2.0 and 
                adx_12h[i] > 25):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low with volume confirmation AND ADX > 25
            elif (close[i] < donchian_low[i] and 
                  vol_ratio > 2.0 and 
                  adx_12h[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals