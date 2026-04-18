#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + 1d ADX Trend Filter
Strategy: Enter long when price breaks above 20-period Donchian high with volume
          and daily ADX > 25 (trending market). Enter short when price breaks
          below 20-period Donchian low with volume and daily ADX > 25.
          Exit when price returns to the middle of the Donchian channel.
          Uses daily ADX to filter for trending conditions only, avoiding
          choppy markets where breakouts fail. Designed for low trade frequency
          with clear trend-following edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ADX for trend strength
    # ADX requires +DI, -DI, and DX calculations
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
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channel (20-period) on 4h data
    def donchian_channel(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_high, donch_low = donchian_channel(high, low, 20)
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            np.isnan(donch_mid[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        mid = donch_mid[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and ADX > 25
            if (price > upper and volume_spike[i] and adx_val > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and ADX > 25
            elif (price < lower and volume_spike[i] and adx_val > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price returns to middle of Donchian channel
            if price <= mid:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price returns to middle of Donchian channel
            if price >= mid:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_1dADX"
timeframe = "4h"
leverage = 1.0