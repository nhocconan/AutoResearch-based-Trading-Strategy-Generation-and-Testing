#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume spike confirmation.
Long when price breaks above Donchian upper (20) AND 1d ADX > 25 (strong trend) AND 4h volume > 1.5x 20-bar average volume.
Short when price breaks below Donchian lower (20) AND 1d ADX > 25 (strong trend) AND 4h volume > 1.5x 20-bar average volume.
Exit when price touches Donchian middle (20-period midpoint) or opposite Donchian band.
Uses 1d for ADX trend filter, 4h for execution and Donchian channels/volume confirmation.
Designed to capture strong breakouts in trending markets with volume confirmation and trend filter.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate 1d ADX (14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for index alignment
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value: simple average
            if period < len(data):
                result[period-1] = np.nanmean(data[:period])
            # Rest: Wilder's smoothing
            for i in range(period, len(data)):
                if np.isnan(result[i-1]):
                    result[i] = np.nanmean(data[i-period+1:i+1])
                else:
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            return result
        
        tr_smoothed = wilders_smoothing(tr, period)
        dm_plus_smoothed = wilders_smoothing(dm_plus, period)
        dm_minus_smoothed = wilders_smoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smoothed / (tr_smoothed + 1e-10)
        di_minus = 100 * dm_minus_smoothed / (tr_smoothed + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 4h Donchian channels (20)
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    upper, lower, middle = donchian_channels(high, low, 20)
    
    # Calculate 4h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(middle[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper[i]
        breakout_lower = close[i] < lower[i]
        
        # Exit conditions: touch middle or opposite band
        touch_middle = abs(close[i] - middle[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < lower[i]) or \
                         (position == -1 and close[i] > upper[i])
        
        if position == 0:
            # Long: break above upper with volume confirmation and strong trend (ADX > 25)
            if (breakout_upper and volume_confirmed and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below lower with volume confirmation and strong trend (ADX > 25)
            elif (breakout_lower and volume_confirmed and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch middle or break below lower
            if (touch_middle or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch middle or break above upper
            if (touch_middle or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADXTrend_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0