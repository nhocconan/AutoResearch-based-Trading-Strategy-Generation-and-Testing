#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation
# Uses daily ADX to ensure trending market, Donchian breakout for entry, and volume spike
# to confirm strength. Designed to capture strong trends while avoiding choppy markets.
# Target: 20-40 trades/year per symbol.
name = "4h_Donchian20_1dADX14_VolumeConfirm"
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
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothing (Wilder's)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[13] = np.nanmean(dx[14:28]) if len(dx) >= 28 else np.nan
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels (20-period)
    upper_channel = np.zeros_like(high_4h)
    lower_channel = np.zeros_like(low_4h)
    
    for i in range(len(high_4h)):
        if i >= 19:
            upper_channel[i] = np.max(high_4h[i-19:i+1])
            lower_channel[i] = np.min(low_4h[i-19:i+1])
        else:
            upper_channel[i] = np.nan
            lower_channel[i] = np.nan
    
    upper_channel_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    
    # Volume confirmation: 20-period volume average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        upper = upper_channel_aligned[i]
        lower = lower_channel_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = vol_current > 1.5 * vol_ma_val
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + ADX > 25 + volume confirmation
            if price > upper and adx_val > 25 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + ADX > 25 + volume confirmation
            elif price < lower and adx_val > 25 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR ADX < 20
            if price < lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR ADX < 20
            if price > upper or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals