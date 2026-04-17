#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX regime filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Exit when price touches Donchian middle band (mean of upper/lower) OR ADX < 20 (range regime).
Uses 1d for ADX trend regime filter, 4h for Donchian breakout and volume.
Target: 75-200 total trades over 4 years (19-50/year). Donchian provides clear breakout levels,
ADX filter avoids whipsaws in ranging markets, volume confirmation ensures breakout strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            first_valid = ~np.isnan(data)
            if np.any(first_valid):
                first_idx = np.where(first_valid)[0][0]
                if first_idx + period < len(data):
                    result[first_idx + period - 1] = np.nanmean(data[first_idx:first_idx + period])
            # Wilder smoothing: today = (prev * (period-1) + current) / period
            for i in range(first_idx + period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period - 1) + data[i]) / period
            return result
        
        atr = wilder_smoothing(tr, period)
        plus_dm_smooth = wilder_smoothing(plus_dm, period)
        minus_dm_smooth = wilder_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = wilder_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels on 4h (20-period)
    lookback = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper_band[i] = np.max(high[i - lookback + 1:i + 1])
        lower_band[i] = np.min(low[i - lookback + 1:i + 1])
        middle_band[i] = (upper_band[i] + lower_band[i]) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = np.full(n, np.nan)
    for i in range(20 - 1, n):
        volume_ma[i] = np.mean(volume[i - 20 + 1:i + 1])
    volume_ratio = volume / (volume_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        price = close[i]
        vol_ratio = volume_ratio[i]
        upper = upper_band[i]
        lower = lower_band[i]
        middle = middle_band[i]
        
        if position == 0:
            # Long: price breaks above upper band AND ADX > 25 (trending) AND volume confirmation
            if price > upper and adx > 25 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND ADX > 25 (trending) AND volume confirmation
            elif price < lower and adx > 25 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches middle band OR ADX < 20 (range regime)
            if price <= middle or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches middle band OR ADX < 20 (range regime)
            if price >= middle or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADXRegime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0