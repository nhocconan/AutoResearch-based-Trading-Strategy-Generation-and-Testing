#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20) + 1d ADX trend filter + volume confirmation.
# In strong trends (ADX>25), Donchian breakouts capture momentum with low false signals.
# Volume > 1.5x average confirms breakout strength. Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14) on daily timeframe
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align length
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value: simple average
            result[period-1] = np.nanmean(arr[1:period])  # skip first NaN
            # Wilder smoothing: new_val = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(arr)):
                if not np.isnan(arr[i]):
                    result[i] = result[i-1] * (1 - 1/period) + arr[i] * (1/period)
                else:
                    result[i] = result[i-1]
            return result
        
        atr = wilder_smooth(tr, period)
        dm_plus_smooth = wilder_smooth(dm_plus, period)
        dm_minus_smooth = wilder_smooth(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Donchian Channel (20) on 12h timeframe
    def donchian_channels(high, low, period):
        upper = np.full(len(high), np.nan)
        lower = np.full(len(low), np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Average volume (20-period = 20*12h = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # start after enough data for indicators
        # Skip if any required data is not ready
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        adx_val = adx_14_1d_aligned[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian + strong trend + volume confirmation
            if price > upper_20[i] and strong_trend and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian + strong trend + volume confirmation
            elif price < lower_20[i] and strong_trend and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian or ADX weakens
            if price < lower_20[i] or adx_val < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian or ADX weakens
            if price > upper_20[i] or adx_val < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_ADX_Volume"
timeframe = "12h"
leverage = 1.0