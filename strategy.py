#!/usr/bin/env python3
"""
Hypothesis: 1-hour mean reversion with 4h Donchian breakout direction and 1d volume confirmation.
In ranging markets (identified by low 4h ADX), look for 1h price rejection at Donchian bands with volume spike.
In trending markets (high 4h ADX), trade breakouts of 1h Donchian channels in direction of 4h trend.
Uses volume filter to avoid false breaks. Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, length=14):
    """Calculate Average Directional Index"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros_like(high)
    for i in range(len(high)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros_like(high)
    for i in range(len(high)):
        if i < length:
            atr[i] = np.nan
        elif i == length:
            atr[i] = np.mean(tr[1:length+1])
        else:
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
    
    plus_di = np.where(atr != 0, 100 * plus_dm / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm / atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.full_like(dx, np.nan)
    for i in range(len(dx)):
        if i < 2*length - 1:
            adx[i] = np.nan
        elif i == 2*length - 1:
            adx[i] = np.mean(dx[length:2*length])
        else:
            adx[i] = (adx[i-1] * (length-1) + dx[i]) / length
    
    return adx

def calculate_donchian_channels(high, low, length=20):
    """Calculate Donchian channels"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(high, np.nan)
    
    for i in range(len(high)):
        if i < length - 1:
            upper[i] = np.nan
            lower[i] = np.nan
        else:
            upper[i] = np.max(high[i-length+1:i+1])
            lower[i] = np.min(low[i-length+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and ranging filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h ADX for trend/ranging detection
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 4h Donchian channels for structure
    upper_4h, lower_4h = calculate_donchian_channels(high_4h, low_4h, 20)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Calculate 1d volume MA for filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h Donchian for entry timing
    upper_1h, lower_1h = calculate_donchian_channels(high, low, 10)
    
    signals = np.zeros(n)
    position = 0
    size = 0.20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Session filter
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
            
        # Skip if data not ready
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_4h_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        vol_now = volume[i]
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Determine market regime
        is_ranging = adx < 25
        is_trending = adx >= 25
        
        if position == 0:
            if is_ranging:
                # Mean reversion: reject at Donchian bands
                if close[i] <= lower_1h[i] and vol_filter:
                    signals[i] = size
                    position = 1
                elif close[i] >= upper_1h[i] and vol_filter:
                    signals[i] = -size
                    position = -1
            else:  # trending
                # Breakout in trend direction
                if close[i] > upper_1h[i] and close[i] > upper_4h_aligned[i] and vol_filter:
                    signals[i] = size
                    position = 1
                elif close[i] < lower_1h[i] and close[i] < lower_4h_aligned[i] and vol_filter:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit: opposite Donchian touch or volume drying up
            if close[i] >= upper_1h[i] or vol_now < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: opposite Donchian touch or volume drying up
            if close[i] <= lower_1h[i] or vol_now < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_DonchianMeanReversion_4hADX_1dVolume"
timeframe = "1h"
leverage = 1.0