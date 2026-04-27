#!/usr/bin/env python3
"""
4h Donchian Channel Breakout with Volume Spike and 1d ADX Trend Filter.
Long when price breaks above Donchian(20) high + volume spike + ADX > 25.
Short when price breaks below Donchian(20) low + volume spike + ADX > 25.
Exit when price returns to Donchian(20) middle band or ADX < 20.
Designed for 20-40 trades/year with strong trend-following edge in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian Channel (20) on 4h data
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan, dtype=np.float64)
        lower = np.full_like(low, np.nan, dtype=np.float64)
        middle = np.full_like(high, np.nan, dtype=np.float64)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
            middle[i] = (upper[i] + lower[i]) / 2.0
        return upper, lower, middle
    
    dc_upper, dc_lower, dc_middle = donchian_channels(high, low, 20)
    
    # Volume filter: volume > 2.0x average (to avoid false breakouts)
    vol_ma_20 = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian(20) + volume MA(20) + daily ADX
    start_idx = max(19, 19, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(dc_middle[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        dc_up = dc_upper[i]
        dc_low = dc_lower[i]
        dc_mid = dc_middle[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        # ADX trend filters
        adx_strong = adx_val > 25.0   # Strong trend for entry
        adx_weak = adx_val < 20.0     # Weak trend for exit
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike + strong ADX
            if price_now > dc_up and vol_filter and adx_strong:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower + volume spike + strong ADX
            elif price_now < dc_low and vol_filter and adx_strong:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle OR ADX weakens
            if price_now < dc_mid or adx_weak:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle OR ADX weakens
            if price_now > dc_mid or adx_weak:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_20_Volume_ADX25_Trend"
timeframe = "4h"
leverage = 1.0