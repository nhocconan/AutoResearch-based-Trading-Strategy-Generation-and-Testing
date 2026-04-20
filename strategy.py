#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w ADX trend filter and volume confirmation
# - Entry: Price breaks above/below Donchian(20) channel + 1w ADX > 25 (trending) + volume > 1.5x average
# - Exit: Price crosses back through Donchian(10) channel or ADX drops below 20
# - Position size: 0.25 long/short
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for ADX calculation (trend filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1w timeframe
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(arr, period):
        smoothed = np.full_like(arr, np.nan)
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr_1w = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Avoid division by zero
    dm_plus_smooth = np.where(atr_1w == 0, 0, dm_plus_smooth)
    dm_minus_smooth = np.where(atr_1w == 0, 0, dm_minus_smooth)
    
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    adx_1w = wilder_smooth(dx, 14)
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Load 1d data for volume average (confirmation)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate Donchian channels on 12h (primary timeframe)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Donchian(20) for entry
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exit (more sensitive)
    highest_high_10 = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if NaN in indicators
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(highest_high_10[i]) or np.isnan(lowest_low_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        volume = volume_12h[i]
        adx = adx_1w_aligned[i]
        vol_avg = volume_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume > 1.5 * vol_avg if not np.isnan(vol_avg) else False
        
        if position == 0:
            # Long entry: Price breaks above Donchian(20) high + ADX > 25 + volume confirmation
            if price > highest_high_20[i] and adx > 25 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian(20) low + ADX > 25 + volume confirmation
            elif price < lowest_low_20[i] and adx > 25 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Donchian(10) low OR ADX drops below 20
            if price < lowest_low_10[i] or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Donchian(10) high OR ADX drops below 20
            if price > highest_high_10[i] or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_1wADX_VolumeFilter"
timeframe = "12h"
leverage = 1.0