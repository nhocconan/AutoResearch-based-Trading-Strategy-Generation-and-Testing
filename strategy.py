#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d ADX trend filter
# Donchian(20) provides objective breakout levels with historical edge in crypto.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# 1d ADX > 25 filters for trending markets only, avoiding choppy losses.
# Designed for 4h timeframe targeting 25-40 trades/year with robust performance in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for trend strength
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr14 = smooth_wilder(tr, 14)
    plus_dm14 = smooth_wilder(plus_dm, 14)
    minus_dm14 = smooth_wilder(minus_dm, 14)
    
    # DI and DX
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    
    # ADX = smoothed DX
    adx_14 = smooth_wilder(dx, 14)
    adx_1d = adx_14  # 14-period ADX
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian(20) channels on 4h data
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper + ADX > 25 + volume confirmation
            if (close[i] > donchian_upper[i] and 
                adx_1d_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + ADX > 25 + volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or ADX falls below 20
            if position == 1:
                # Exit long: price returns below Donchian lower or weak trend
                if (close[i] < donchian_lower[i] or 
                    adx_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns above Donchian upper or weak trend
                if (close[i] > donchian_upper[i] or 
                    adx_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX25_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0