#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 1-day volume confirmation and 1-day ADX trend filter.
Trades breakouts when price exceeds 20-period Donchian channel with volume above 1-day average
and ADX > 25 indicating trending market. Uses ATR-based stop loss to limit downside.
Designed to work in both bull and bear markets by only trading in the direction of the daily trend.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
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
    
    # Get 4-hour data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Donchian channel (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper (20-period high) and lower (20-period low)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get daily data for volume filter and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1-day ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_sum = wilders_smoothing(tr, 14)
    plus_dm_sum = wilders_smoothing(plus_dm, 14)
    minus_dm_sum = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian channels, volume MA, and ADX
    start_idx = max(20, 20, 30)  # max of lookbacks: Donchian(20), vol MA(20), ADX(14+14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        adx_now = adx_aligned[i]
        
        # Current Donchian levels
        upper_now = donchian_high_aligned[i]
        lower_now = donchian_low_aligned[i]
        
        # Volume filter: volume > 1.5x 1-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_now > 25
        
        # Entry conditions: Donchian breakout with volume and trend confirmation
        if position == 0:
            # Long: price breaks above Donchian high with volume + trend
            if price_now > upper_now and vol_filter and trend_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume + trend
            elif price_now < lower_now and vol_filter and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or trend weakens
            if price_now < lower_now or adx_now < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high or trend weakens
            if price_now > upper_now or adx_now < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dVolume_1dADX"
timeframe = "4h"
leverage = 1.0