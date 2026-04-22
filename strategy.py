# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian Channel Breakout with 1-day ADX filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band on 12h, ADX > 25 on 1d (trending), and 1-day volume > 20-period average volume.
Short when price breaks below 20-period Donchian lower band on 12h, ADX > 25 on 1d, and 1-day volume > 20-period average volume.
Exit when price crosses the 10-period EMA on 12h (opposite direction).
Uses institutional volume and trend confirmation to capture sustained moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ADX and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1-day data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ , DM- with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # Skip index 0 (nan)
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    for i in range(27, len(dx)):  # 2*period-1 for ADX
        if i == 27:
            adx[i] = np.nanmean(dx[14:28])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter on 1-day
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Donchian Channel on 12h data (20-period)
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
        return upper, lower
    
    upper_band, lower_band = donchian_channels(high, low, 20)
    
    # 10-period EMA on 12h for exit
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or
            np.isnan(ema_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian band, ADX > 25, volume above average
            if (close[i] > upper_band[i] and 
                adx_aligned[i] > 25 and 
                volume_1d[i] > avg_vol_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band, ADX > 25, volume above average
            elif (close[i] < lower_band[i] and 
                  adx_aligned[i] > 25 and 
                  volume_1d[i] > avg_vol_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 10-period EMA
                if close[i] < ema_10[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 10-period EMA
                if close[i] > ema_10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_1dADX_Volume_Filter"
timeframe = "12h"
leverage = 1.0