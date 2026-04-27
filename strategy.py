#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and ADX Trend Filter.
Long when price breaks above Donchian(20) high + ADX > 25 + volume > 1.5x average.
Short when price breaks below Donchian(20) low + ADX > 25 + volume > 1.5x average.
Exit when price crosses back below Donchian midpoint (long) or above midpoint (short).
Designed to generate 20-50 trades/year per symbol with strong trend-following edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    return tr

def average_true_range(high, low, close, period):
    """Calculate Average True Range"""
    tr = true_range(high, low, close)
    atr = np.empty_like(tr, dtype=np.float64)
    atr.fill(np.nan)
    if len(tr) < period:
        return atr
    # First value is simple average
    atr[period-1] = np.mean(tr[:period])
    # Wilder's smoothing
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def directional_movement(high, low, close, period):
    """Calculate +DI and -DI for ADX"""
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1  # Invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth using Wilder's method (same as ATR)
    tr = true_range(high, low, close)
    atr = np.empty_like(tr, dtype=np.float64)
    atr.fill(np.nan)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Calculate DI
    plus_di = np.full_like(close, np.nan, dtype=np.float64)
    minus_di = np.full_like(close, np.nan, dtype=np.float64)
    
    for i in range(period-1, len(close)):
        if not np.isnan(atr[i]) and atr[i] != 0:
            plus_di[i] = (np.sum(plus_dm[i-period+1:i+1]) / atr[i]) * 100
            minus_di[i] = (np.sum(minus_dm[i-period+1:i+1]) / atr[i]) * 100
    
    return plus_di, minus_di

def adx(high, low, close, period):
    """Calculate Average Directional Index"""
    plus_di, minus_di = directional_movement(high, low, close, period)
    
    # Calculate DX
    dx = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(len(close)):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = np.abs(plus_di[i] - minus_di[i]) / di_sum * 100
    
    # Smooth DX to get ADX
    adx_val = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(period-1, len(close)):
        if i == period-1:
            # First ADX is average of first 'period' DX values
            valid_dx = dx[period-1:2*period-1][~np.isnan(dx[period-1:2*period-1])]
            if len(valid_dx) > 0:
                adx_val[i] = np.mean(valid_dx)
        elif i >= period:
            if not np.isnan(dx[i]) and not np.isnan(adx_val[i-1]):
                adx_val[i] = (adx_val[i-1] * (period-1) + dx[i]) / period
    
    return adx_val

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX (14-period) and Donchian (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # ADX calculation
    adx_val = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_val)
    
    # Donchian channels (20-period high/low)
    donch_high = np.empty_like(df_1d['close'].values, dtype=np.float64)
    donch_high.fill(np.nan)
    donch_low = np.empty_like(df_1d['close'].values, dtype=np.float64)
    donch_low.fill(np.nan)
    
    for i in range(19, len(df_1d)):
        donch_high[i] = np.max(df_1d['high'].values[i-19:i+1])
        donch_low[i] = np.min(df_1d['low'].values[i-19:i+1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid = (donch_high_aligned + donch_low_aligned) / 2.0
    
    # Volume filter: volume > 1.5x average (20-period)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ADX (14) + Donchian (20) + volume MA (20)
    start_idx = max(14, 19, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicator values
        adx_now = adx_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        donch_mid_val = donch_mid[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        # Trend filter: ADX > 25
        trend_filter = adx_now > 25
        
        if position == 0:
            # Long: price breaks above Donchian high + trend + volume
            if price_now > donch_high_val and trend_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + trend + volume
            elif price_now < donch_low_val and trend_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if price_now < donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if price_now > donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0