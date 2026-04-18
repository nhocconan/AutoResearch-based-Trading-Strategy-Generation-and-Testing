#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout with 1w volume confirmation and 1w ADX trend filter.
- Long: Price breaks above Donchian(20) high, weekly volume > 1.5x 20-week average, ADX > 25
- Short: Price breaks below Donchian(20) low, weekly volume > 1.5x 20-week average, ADX > 25
- Exit: Price crosses back through Donchian midpoint or volume drops below 1.2x average
Designed for 7-25 trades/year (30-100 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands."""
    upper = np.full(len(high), np.nan)
    lower = np.full(len(low), np.nan)
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    return upper, lower

def calculate_sma(arr, period):
    """Calculate Simple Moving Average."""
    sma = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        sma[i] = np.mean(arr[i-period+1:i+1])
    return sma

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index."""
    if len(high) < period * 2:
        return np.full(len(high), np.nan)
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
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
    
    # Calculate Directional Indicators
    plus_di = np.full(len(dm_plus), np.nan)
    minus_di = np.full(len(dm_minus), np.nan)
    for i in range(period, len(atr)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
    
    # Calculate DX and ADX
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
    
    # Get 1w data for volume and ADX
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channel (20-period) on 1d
    donch_h, donch_l = calculate_donchian(high, low, 20)
    donch_mid = (donch_h + donch_l) / 2
    
    # Calculate volume moving average (20-period) on 1w
    vol_ma_1w = calculate_sma(volume_1w, 20)
    
    # Calculate ADX (14-period) on 1w
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align 1w indicators to 1d timeframe
    vol_ma_1w_1d = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    adx_14_1w_1d = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need Donchian, volume MA, and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_h[i]) or np.isnan(donch_l[i]) or 
            np.isnan(vol_ma_1w_1d[i]) or np.isnan(adx_14_1w_1d[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1w volume for current 1d bar
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        
        # Volume confirmation: current 1w volume > 1.5x 20-week average
        vol_spike = vol_1w_aligned[i] > 1.5 * vol_ma_1w_1d[i]
        
        if position == 0:
            # Long: break above upper band, volume spike, ADX > 25
            if close[i] > donch_h[i] and vol_spike and adx_14_1w_1d[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band, volume spike, ADX > 25
            elif close[i] < donch_l[i] and vol_spike and adx_14_1w_1d[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint or volume drops below 1.2x average
            vol_exit = vol_1w_aligned[i] < 1.2 * vol_ma_1w_1d[i]
            if close[i] < donch_mid[i] or vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint or volume drops below 1.2x average
            vol_exit = vol_1w_aligned[i] < 1.2 * vol_ma_1w_1d[i]
            if close[i] > donch_mid[i] or vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_VolumeSpike_ADX14_1w"
timeframe = "1d"
leverage = 1.0