#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Uses 1d ADX > 25 to identify trending markets (works in both bull/bear regimes)
# Donchian(20) from prior 6h session provides clear breakout levels
# Volume confirmation (>2.0x 20 EMA) filters low-participation false breakouts
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# ADX filter reduces whipsaws in ranging markets while capturing strong trends.

name = "6h_Donchian20_1dADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilders_smoothing(values, alpha):
        smoothed = np.full_like(values, np.nan, dtype=float)
        smoothed[period] = np.nansum(values[1:period+1])  # first value: simple average
        for i in range(period+1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    tr_smoothed = wilders_smoothing(tr, alpha)
    plus_dm_smoothed = wilders_smoothing(plus_dm, alpha)
    minus_dm_smoothed = wilders_smoothing(minus_dm, alpha)
    
    # Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # Directional Index (DX) and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, alpha)
    
    # Align 1d ADX to 6h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) from previous 6h bar
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Rolling max/min for Donchian channels
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high_6h, 20)
    donchian_low = rolling_min(low_6h, 20)
    
    # Align Donchian levels to 6h timeframe (completed 6h bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trending = adx_aligned[i] > 25.0
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + trending + volume spike
            if close[i] > donchian_high_aligned[i] and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + trending + volume spike
            elif close[i] < donchian_low_aligned[i] and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend weakens OR volume drops
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            
            if (close[i] < donchian_mid or 
                adx_aligned[i] < 20.0 or  # trend weakening
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend weakens OR volume drops
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            
            if (close[i] > donchian_mid or 
                adx_aligned[i] < 20.0 or  # trend weakening
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals