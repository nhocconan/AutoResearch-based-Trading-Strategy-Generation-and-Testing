#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume spike and 1w ADX trend filter
# Uses Donchian(20) breakout on 12h for trend direction, confirmed by
# 1d volume spike (>1.5x 20-period average) and 1w ADX > 25 for trend strength.
# Designed to capture strong trends with low frequency to minimize fee drag.
# Target: 15-30 trades/year on 12h timeframe.
name = "12h_Donchian20_1dVolumeSpike_1wADX_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels (20-period high/low)
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-20:i])
    
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_spike = volume_1d > (vol_ma_20 * 1.5)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and Directional Movement
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 30)
    dm_plus_smooth = wilders_smoothing(dm_plus, 30)
    dm_minus_smooth = wilders_smoothing(dm_minus, 30)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 30)
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # Convert back to boolean
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND volume spike AND ADX > 25
            if price > donch_high and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND volume spike AND ADX > 25
            elif price < donch_low and vol_spike and adx_val > 25:
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