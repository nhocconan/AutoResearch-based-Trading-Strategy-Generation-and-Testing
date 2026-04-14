#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Williams Alligator to define trend direction (Jaws/Teeth/Lips),
# entering long when price > Lips and Lips > Teeth > Jaws (bullish alignment),
# entering short when price < Lips and Lips < Teeth < Jaws (bearish alignment),
# with 12h volume confirmation (>1.5x 20-period average) and ADX(12h) > 25 for trend strength.
# Exits when Alligator alignment breaks or ADX < 20.
# Williams Alligator uses smoothed medians (SMMA) to avoid whipsaws in sideways markets.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag while capturing strong trends.

def calculate_smma(data, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    result = np.full_like(data, np.nan)
    if len(data) < period:
        return result
    # First value: simple average
    result[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (13,8,5) - Smoothed Medians
    # Jaw: 13-period SMMA of median, shifted 8 bars
    # Teeth: 8-period SMMA of median, shifted 5 bars
    # Lips: 5-period SMMA of median, shifted 3 bars
    median_1d = (high_1d + low_1d) / 2.0
    
    jaw_raw = calculate_smma(median_1d, 13)
    teeth_raw = calculate_smma(median_1d, 8)
    lips_raw = calculate_smma(median_1d, 5)
    
    # Apply shifts (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    jaw[8:] = jaw_raw[:-8] if len(jaw_raw) > 8 else np.nan
    teeth[5:] = teeth_raw[:-5] if len(teeth_raw) > 5 else np.nan
    lips[3:] = lips_raw[:-3] if len(lips_raw) > 3 else np.nan
    
    # Load 12h data ONCE for ADX and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ADX (14-period)
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average (skip first element for DM)
        result[period-1] = np.nanmean(data[1:period])  # Skip first element which is 0 for DM
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.full_like(dx, np.nan)
    # First ADX: simple average of first 14 DX values
    valid_dx = dx[~np.isnan(dx)]
    if len(valid_dx) >= 14:
        adx[13] = np.mean(valid_dx[:14])
        for i in range(14, len(dx)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate 20-period average volume for 12h
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(19, len(volume_12h)):
        vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(40, 20)  # Need 1d and 12h data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        volume_ratio = volume_12h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        # Check Alligator alignment
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] and 
                            teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] and 
                            teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Look for entries with volume confirmation and trend
            # Long: bullish alignment AND price > Lips AND volume > 1.5x average AND ADX > 25
            if (bullish_alignment and 
                close[i] > lips_aligned[i] and 
                volume_ratio > 1.5 and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: bearish alignment AND price < Lips AND volume > 1.5x average AND ADX > 25
            elif (bearish_alignment and 
                  close[i] < lips_aligned[i] and 
                  volume_ratio > 1.5 and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: alignment breaks or trend weakens
            if (not bullish_alignment or 
                close[i] < lips_aligned[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: alignment breaks or trend weakens
            if (not bearish_alignment or 
                close[i] > lips_aligned[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Williams_Alligator_Volume_ADX_v1"
timeframe = "12h"
leverage = 1.0