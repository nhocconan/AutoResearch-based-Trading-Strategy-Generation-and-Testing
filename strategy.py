#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation
# Donchian channels identify breakout points with clear support/resistance levels.
# 1d ADX > 25 ensures trading only in trending markets, reducing whipsaw in chop.
# Volume > 1.5x 20-period average confirms institutional participation in breakouts.
# Designed for 6h timeframe targeting 15-30 trades/year with strong performance in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate 1d ADX(14) for trend strength
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_series(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # Initial value
        result[period-1] = np.nansum(data[:period])
        # Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(atr, np.nan, dtype=float)
    di_minus = np.full_like(atr, np.nan, dtype=float)
    valid = (~np.isnan(atr)) & (atr != 0)
    di_plus[valid] = (dm_plus_smooth[valid] / atr[valid]) * 100
    di_minus[valid] = (dm_minus_smooth[valid] / atr[valid]) * 100
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan, dtype=float)
    di_sum = di_plus + di_minus
    valid_dx = (~np.isnan(di_plus)) & (~np.isnan(di_minus)) & (di_sum != 0)
    dx[valid_dx] = (np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]) * 100
    
    adx = smooth_series(dx, 14)
    adx_25 = adx > 25  # Trend strength threshold
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_25)
    
    # Donchian(20) channels on 6h data
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan, dtype=float)
        lower = np.full_like(high, np.nan, dtype=float)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        vol_avg_20[i] = np.mean(volume[i-19:i+1])
    vol_threshold = 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper + ADX trend + volume confirmation
            if (close[i] > dc_upper[i] and 
                adx_aligned[i] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + ADX trend + volume confirmation
            elif (close[i] < dc_lower[i] and 
                  adx_aligned[i] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend weakens
            if position == 1:
                # Exit long: price returns below Donchian lower or ADX < 20
                if (close[i] < dc_lower[i] or 
                    not adx_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns above Donchian upper or ADX < 20
                if (close[i] > dc_upper[i] or 
                    not adx_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dADX25_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0