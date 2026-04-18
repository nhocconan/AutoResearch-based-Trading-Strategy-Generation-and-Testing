#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Spike and 1d ADX Trend Filter
Long when price breaks above Donchian(20) high with volume spike and 1d ADX > 25
Short when price breaks below Donchian(20) low with volume spike and 1d ADX > 25
Uses 1d ADX as higher timeframe trend filter to ensure trending market conditions.
Designed for low trade frequency with clear trend-following edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for 1d ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Rest is Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d ADX > 25 indicates trending market
        strong_trend = adx_1d_aligned[i] > 25
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and strong trend
            if (price > donchian_high[i] and volume_spike[i] and strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and strong trend
            elif (price < donchian_low[i] and volume_spike[i] and strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below Donchian low or trend weakens
            if (price < donchian_low[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above Donchian high or trend weakens
            if (price > donchian_high[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_1dADX_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0