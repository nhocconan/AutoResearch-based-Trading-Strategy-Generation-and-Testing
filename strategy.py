#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above Donchian upper band with volume > 1.5x average and ADX > 25 (trending).
Short when price breaks below Donchian lower band with volume > 1.5x average and ADX > 25.
Exit when price reverts to Donchian midpoint or ADX < 20 (trend weakens).
Uses 12h for price/volume/Dochian, 1d for ADX filter.
Target: 50-150 total trades over 4 years (12-37/year). Focus on strong trends with volume confirmation to avoid chop losses.
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        mid = np.full_like(high, np.nan)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
            mid[i] = (upper[i] + lower[i]) / 2.0
        return upper, lower, mid
    
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(high)
        dm_minus = np.zeros_like(high)
        for i in range(1, len(high)):
            dm_plus[i] = max(high[i] - high[i-1], 0)
            dm_minus[i] = max(low[i-1] - low[i], 0)
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        tr_period = np.zeros_like(high)
        dm_plus_period = np.zeros_like(high)
        dm_minus_period = np.zeros_like(high)
        
        # Initial values
        tr_period[period] = np.mean(tr[1:period+1])
        dm_plus_period[period] = np.mean(dm_plus[1:period+1])
        dm_minus_period[period] = np.mean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(high)):
            tr_period[i] = (tr_period[i-1] * (period-1) + tr[i]) / period
            dm_plus_period[i] = (dm_plus_period[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_period[i] = (dm_minus_period[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros_like(high)
        di_minus = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if tr_period[i] > 0:
                di_plus[i] = 100 * dm_plus_period[i] / tr_period[i]
                di_minus[i] = 100 * dm_minus_period[i] / tr_period[i]
                if (di_plus[i] + di_minus[i]) > 0:
                    dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
                else:
                    dx[i] = 0
        
        # ADX (smoothed DX)
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period]) if 2*period <= len(high) else 0
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, 20)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_conf = volume_confirm[i]
        adx_val = adx_1d_aligned[i]
        upper = donch_upper[i]
        lower = donch_lower[i]
        mid = donch_mid[i]
        
        # Trend regime: ADX > 25 = strong trend (good for breakout)
        is_trending = adx_val > 25
        # Weak trend: ADX < 20 = trend weakening (exit)
        is_weak_trend = adx_val < 20
        
        if position == 0:
            # Long: price breaks above upper band with volume confirmation in strong trend
            if price > upper and vol_conf and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume confirmation in strong trend
            elif price < lower and vol_conf and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR trend weakens
            if price <= mid or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR trend weakens
            if price >= mid or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX_Volume_Confirm"
timeframe = "12h"
leverage = 1.0