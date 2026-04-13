#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Price Channel Breakout (Donchian 10) with 12h ADX trend filter and volume confirmation.
# Uses Donchian channel breakouts for entry, filtered by 12h ADX > 25 to ensure trending markets,
# and volume > 1.5x average for confirmation. Exits when price crosses the opposite Donchian band.
# Designed to work in both bull and bear markets by only trading strong trends.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for ADX calculation
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        valid_start = ~np.isnan(arr)
        if not np.any(valid_start):
            return result
        first_valid = np.where(valid_start)[0][0]
        if first_valid + period >= len(arr):
            return result
        result[first_valid + period - 1] = np.nanmean(arr[first_valid:first_valid + period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(first_valid + period, len(arr)):
            if np.isnan(arr[i]):
                result[i] = result[i-1]
            else:
                result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
        return result
    
    period = 10
    atr = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period)
    
    # Align 12h ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Donchian Channel (10-period) on 4h timeframe
    def rolling_max(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_channel = rolling_max(high, 10)
    lower_channel = rolling_min(low, 10)
    
    # Average volume (20-period = 20*4h = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian channel + trend + volume
            if (price > upper_channel[i] and 
                trend_filter and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian channel + trend + volume
            elif (price < lower_channel[i] and 
                  trend_filter and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian channel
            if price < lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian channel
            if price > upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_ADX_Volume"
timeframe = "4h"
leverage = 1.0