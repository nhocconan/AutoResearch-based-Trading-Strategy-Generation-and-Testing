#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and 1d ADX Trend Filter.
Long when: 1) Price breaks above Donchian(20) upper band, 2) Volume > 2x 20-period average, 3) 1d ADX > 25 (strong trend).
Short when: 1) Price breaks below Donchian(20) lower band, 2) Volume > 2x 20-period average, 3) 1d ADX > 25 (strong trend).
Exit when price returns to middle band (mean reversion) or ADX < 20 (weak trend).
Designed for 4h timeframe: targets 75-200 total trades over 4 years (19-50/year).
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
    
    # Donchian(20) channels
    donchian_window = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    
    for i in range(donchian_window - 1, n):
        upper_band[i] = np.max(high[i-donchian_window+1:i+1])
        lower_band[i] = np.min(low[i-donchian_window+1:i+1])
        middle_band[i] = (upper_band[i] + lower_band[i]) / 2.0
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily timeframe
    adx_period = 14
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First element is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = np.full(len(tr), np.nan)
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    
    # Initial values (simple average)
    if len(tr) >= adx_period:
        atr[adx_period-1] = np.nanmean(tr[1:adx_period])  # Skip first NaN
        dm_plus_smooth[adx_period-1] = np.nanmean(dm_plus[1:adx_period])
        dm_minus_smooth[adx_period-1] = np.nanmean(dm_minus[1:adx_period])
        
        # Wilder's smoothing
        for i in range(adx_period, len(tr)):
            atr[i] = (atr[i-1] * (adx_period-1) + tr[i]) / adx_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (adx_period-1) + dm_plus[i]) / adx_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (adx_period-1) + dm_minus[i]) / adx_period
    
    # DI+ and DI-
    di_plus = np.full(len(tr), np.nan)
    di_minus = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
    
    dx_sum = di_plus + di_minus
    dx_valid = valid & (dx_sum != 0)
    dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / dx_sum[dx_valid]
    
    # ADX: smoothed DX
    adx = np.full(len(tr), np.nan)
    if len(dx) >= adx_period:
        # First ADX value is average of first 'adx_period' DX values
        first_adx_idx = adx_period - 1
        if first_adx_idx < len(dx):
            valid_dx = dx[1:adx_period+1]  # Skip first NaN in DX
            if len(valid_dx) == adx_period:
                adx[first_adx_idx] = np.nanmean(valid_dx)
        
        # Subsequent ADX values: Wilder's smoothing
        for i in range(first_adx_idx + 1, len(dx)):
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian(20), volume MA(20), ADX(14) with smoothing
    start_idx = max(donchian_window, 19, 30)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        middle = middle_band[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        # ADX filters: strong trend (>25) for entry, weak trend (<20) for exit
        strong_trend = adx_val > 25
        weak_trend = adx_val < 20
        
        if position == 0:
            # Long: price breaks above upper band + volume spike + strong trend
            if price > upper and vol_filter and strong_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band + volume spike + strong trend
            elif price < lower and vol_filter and strong_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band or trend weakens
            if price <= middle or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle band or trend weakens
            if price >= middle or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX_Trend"
timeframe = "4h"
leverage = 1.0