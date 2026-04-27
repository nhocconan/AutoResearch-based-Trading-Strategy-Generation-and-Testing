#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and Trend Filter.
Long when price breaks above weekly Donchian high (20) and price > weekly EMA34.
Short when price breaks below weekly Donchian low (20) and price < weekly EMA34.
Exit when price crosses back below weekly EMA34 (long) or above EMA34 (short).
Designed to generate 15-25 trades/year per symbol with strong trend-following edge.
Works in both bull and bear markets by using weekly timeframe for trend and entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(arr, period):
    """Exponential Moving Average with proper initialization"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result.fill(np.nan)
    if n < period:
        return result
    # Initialize first value as SMA
    result[period-1] = np.mean(arr[:period])
    # Calculate EMA for remaining values
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and EMA
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period high/low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    donchian_high = np.empty_like(weekly_high, dtype=np.float64)
    donchian_low = np.empty_like(weekly_low, dtype=np.float64)
    donchian_high.fill(np.nan)
    donchian_low.fill(np.nan)
    
    for i in range(19, len(weekly_high)):
        donchian_high[i] = np.max(weekly_high[i-19:i+1])
        donchian_low[i] = np.min(weekly_low[i-19:i+1])
    
    # Weekly EMA34 on close
    weekly_close = df_weekly['close'].values
    weekly_ema34 = ema(weekly_close, 34)
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema34)
    
    # Volume filter: volume > 1.3x 20-day average (to avoid false signals)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA (34) + Donchian (20) + volume MA (20)
    start_idx = max(34, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_ema34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current weekly indicator values
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema34_val = weekly_ema34_aligned[i]
        
        # Volume filter: volume > 1.3x average
        vol_filter = vol_now > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high AND price > weekly EMA34 + volume filter
            if price_now > donchian_high_val and price_now > ema34_val and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly Donchian low AND price < weekly EMA34 + volume filter
            elif price_now < donchian_low_val and price_now < ema34_val and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below weekly EMA34
            if price_now < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above weekly EMA34
            if price_now > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0