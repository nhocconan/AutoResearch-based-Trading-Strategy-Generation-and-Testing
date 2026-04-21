#!/usr/bin/env python3
"""
6h strategy: Donchian(20) breakout with 12h volume confirmation and 1d ADX trend filter.
Long when price breaks above Donchian upper (20-period high) with volume spike (>1.5x) and 1d ADX > 25.
Short when price breaks below Donchian lower (20-period low) with volume spike and 1d ADX > 25.
Exit when price crosses the 20-period moving average (middle of Donchian channel).
Volume confirmation reduces false breakouts; ADX ensures we only trade in trending markets.
Works in bull markets (buy breakouts) and bear markets (sell breakdowns). Target: 12-37 trades/year for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channel (highest high, lowest low)
    high_max = prices['high'].rolling(window=20, min_periods=20).max().values
    low_min = prices['low'].rolling(window=20, min_periods=20).min().values
    ma_20 = ((high_max + low_min) / 2.0)  # Middle line for exit
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # 12h volume ratio (current volume / 20-period average)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = vol_12h / vol_ma_20_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Rest is Wilder smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ma_20[i]) or np.isnan(vol_ratio_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio_12h_aligned[i]
        adx_val = adx_1d_aligned[i]
        vol_threshold = 1.5  # Volume spike threshold
        adx_threshold = 25.0  # ADX trend threshold
        
        if position == 0:
            # Enter long: price breaks above Donchian upper + volume spike + ADX > 25
            if (price_close > high_max[i] and 
                vol_ratio_val > vol_threshold and 
                adx_val > adx_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower + volume spike + ADX > 25
            elif (price_close < low_min[i] and 
                  vol_ratio_val > vol_threshold and 
                  adx_val > adx_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses 20-period moving average (middle of Donchian)
            if position == 1 and price_close < ma_20[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_DonchianBreakout_12hVolume_1dADX"
timeframe = "6h"
leverage = 1.0