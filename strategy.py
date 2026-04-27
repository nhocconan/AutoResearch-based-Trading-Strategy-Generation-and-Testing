#!/usr/bin/env python3
"""
12h Donchian Channel Breakout with Daily Volume Spike and ADX Trend Filter.
Long when price breaks above 20-period Donchian upper band + ADX > 25 + volume spike.
Short when price breaks below 20-period Donchian lower band + ADX > 25 + volume spike.
Exit when price crosses back through the middle of the Donchian channel.
Designed for low frequency (12-37 trades/year) to minimize fee drag.
Works in both bull and bear markets by capturing strong trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, period)
    dmp = smooth_wilder(dm_plus, period)
    dmm = smooth_wilder(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dmp / atr, 0)
    di_minus = np.where(atr != 0, 100 * dmm / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe for trend strength
    adx_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d = align_htf_to_ltf(prices, df_1d, adx_raw)
    
    # Calculate Donchian channels on 12h timeframe (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i-donchian_period+1:i+1])
        lower[i] = np.min(low[i-donchian_period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # Volume filter: volume > 1.8x average (to avoid false breakouts)
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), ADX (14+14), volume MA (20)
    start_idx = max(donchian_period - 1, 27, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(adx_1d[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        upper_band = upper[i]
        lower_band = lower[i]
        middle_band = middle[i]
        adx_val = adx_1d[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_ma_20[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian band + strong trend + volume spike
            if price_now > upper_band and trend_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian band + strong trend + volume spike
            elif price_now < lower_band and trend_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below middle of Donchian channel
            if price_now < middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above middle of Donchian channel
            if price_now > middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DonchianBreakout_ADXTrend_Volume"
timeframe = "12h"
leverage = 1.0