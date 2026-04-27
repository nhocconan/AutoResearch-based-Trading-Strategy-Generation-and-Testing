#!/usr/bin/env python3
"""
6h Williams Alligator + 1-day ADX Trend Filter + Volume Spike
Long when: Alligator bullish (Jaw < Teeth < Lips), ADX > 25 (trending), volume > 2x MA20
Short when: Alligator bearish (Jaw > Teeth > Lips), ADX > 25, volume > 2x MA20
Williams Alligator identifies trend direction and alignment, ADX filters for trending markets,
volume confirms institutional participation. Designed for 6H timeframe to avoid overtrading.
Target: 20-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.full_like(values, np.nan, dtype=np.float64)
        if len(values) < period:
            return smoothed
        # First value is simple average
        smoothed[period-1] = np.nanmean(values[1:period])
        # Subsequent values: smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        for i in range(period, len(values)):
            if not np.isnan(smoothed[i-1]) and not np.isnan(values[i]):
                smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
            else:
                smoothed[i] = np.nan
        return smoothed
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = np.full_like(tr_smoothed, np.nan, dtype=np.float64)
    di_minus = np.full_like(tr_smoothed, np.nan, dtype=np.float64)
    dx = np.full_like(tr_smoothed, np.nan, dtype=np.float64)
    
    for i in range(len(tr_smoothed)):
        if not np.isnan(tr_smoothed[i]) and tr_smoothed[i] != 0:
            di_plus[i] = (dm_plus_smoothed[i] / tr_smoothed[i]) * 100
            di_minus[i] = (dm_minus_smoothed[i] / tr_smoothed[i]) * 100
            if not np.isnan(di_plus[i]) and not np.isnan(di_minus[i]):
                dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    # ADX: smoothed DX
    adx = wilders_smoothing(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator (6-period, 5-period, 3-period SMAs with future shifts)
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    def smoothed_ma(values, period):
        """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
        sma = np.full_like(values, np.nan, dtype=np.float64)
        if len(values) < period:
            return sma
        # First value is simple average
        sma[period-1] = np.mean(values[:period])
        # Subsequent values: sma[i] = (sma[i-1] * (period-1) + values[i]) / period
        for i in range(period, len(values)):
            sma[i] = (sma[i-1] * (period-1) + values[i]) / period
        return sma
    
    jaw_raw = smoothed_ma(close, 13)
    teeth_raw = smoothed_ma(close, 8)
    lips_raw = smoothed_ma(close, 5)
    
    # Shift forward: Jaw(+8), Teeth(+5), Lips(+3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    for i in range(len(jaw)):
        if i + 8 < len(jaw) and not np.isnan(jaw_raw[i]):
            jaw[i + 8] = jaw_raw[i]
    for i in range(len(teeth)):
        if i + 5 < len(teeth) and not np.isnan(teeth_raw[i]):
            teeth[i + 5] = teeth_raw[i]
    for i in range(len(lips)):
        if i + 3 < len(lips) and not np.isnan(lips_raw[i]):
            lips[i + 3] = lips_raw[i]
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume filter: volume > 2x average (calculated from 6h volume MA20)
    vol_ma_20 = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator components (13+8=21 for teeth, plus shifts)
    start_idx = 21  # Need enough data for SMAs and shifts
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_1d_aligned[i]
        
        # Alligator signals
        alligator_bullish = (jaw_val < teeth_val) and (teeth_val < lips_val)
        alligator_bearish = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # ADX trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull conditions: Alligator bullish + strong trend + volume
            if alligator_bullish and strong_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear conditions: Alligator bearish + strong trend + volume
            elif alligator_bearish and strong_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish or trend weakens
            if not alligator_bullish or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator turns bullish or trend weakens
            if not alligator_bearish or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsAlligator_1dADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0