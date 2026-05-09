#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Designed to capture strong trends while avoiding false breakouts in ranging markets.
# Uses: 4h price breaking 20-period Donchian channel, 1d ADX > 25 for trend strength,
# and volume > 1.5x 20-period average for confirmation. Target: 25-40 trades/year.
name = "4h_Donchian20_1dADX25_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel calculation (same timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channel on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian high: 20-period rolling max
    donchian_high = np.full(len(high_4h), np.nan)
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
    
    # Donchian low: 20-period rolling min
    donchian_low = np.full(len(low_4h), np.nan)
    for i in range(20, len(low_4h)):
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-period ADX on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.full_like(values, np.nan)
        if len(values) < period:
            return smoothed
        # First value is simple average
        smoothed[period-1] = np.nanmean(values[1:period])
        # Subsequent values
        for i in range(period, len(values)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        return smoothed
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio = volume_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + ADX > 25 + volume confirmation
            if price > donch_high and adx_val > 25 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + ADX > 25 + volume confirmation
            elif price < donch_low and adx_val > 25 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR ADX < 20 (trend weakening)
            if price < donch_low or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR ADX < 20 (trend weakening)
            if price > donch_high or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals