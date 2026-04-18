#!/usr/bin/env python3
"""
1d Donchian Breakout + Volume Spike + Weekly ADX Filter
Long: Price breaks above Donchian(20) high + volume spike + weekly ADX > 25
Short: Price breaks below Donchian(20) low + volume spike + weekly ADX > 25
Exit: Price crosses opposite Donchian band or ADX drops below 20
Designed for 1d timeframe to capture strong trending moves with volume confirmation
and trend strength filter. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, window):
    """Calculate Donchian channels: upper and lower bands"""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    return upper, lower

def calculate_adx(high, low, close, window):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/window)
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    atr[window-1] = np.mean(tr[:window])
    dm_plus_smooth[window-1] = np.mean(dm_plus[:window])
    dm_minus_smooth[window-1] = np.mean(dm_minus[:window])
    
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window-1) + tr[i]) / window
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (window-1) + dm_plus[i]) / window
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (window-1) + dm_minus[i]) / window
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    adx[2*window-1] = np.mean(dx[window:2*window])
    
    for i in range(2*window, len(dx)):
        adx[i] = (adx[i-1] * (window-1) + dx[i]) / window
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels (20-period)
    donch_high, donch_low = calculate_donchian(high, low, 20)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get weekly data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX (14-period)
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align weekly ADX to daily
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need Donchian and ADX calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume spike + ADX > 25
            if (price > donch_high[i] and 
                volume_spike[i] and 
                adx_14_1w_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume spike + ADX > 25
            elif (price < donch_low[i] and 
                  volume_spike[i] and 
                  adx_14_1w_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Donchian low OR ADX drops below 20
            if (price < donch_low[i]) or (adx_14_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Donchian high OR ADX drops below 20
            if (price > donch_high[i]) or (adx_14_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_VolumeSpike_WeeklyADX"
timeframe = "1d"
leverage = 1.0