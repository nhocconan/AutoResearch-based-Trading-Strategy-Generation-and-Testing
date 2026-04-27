#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal strategy with 1-day volume spike filter and ADX trend filter.
# Long when price touches S3 level with volume spike and ADX > 25 (trending market).
# Short when price touches R3 level with volume spike and ADX > 25.
# Exit when price touches S1 (for longs) or R1 (for shorts).
# Uses Camarilla levels from daily pivot (high/low/close) for precise reversal zones.
# Target: 20-40 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point and Camarilla levels
    # Pivot = (High + Low + Close) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # S3 = Close - (Range * 1.100 / 6)
    s3 = close_1d - (range_1d * 1.100 / 6)
    # S2 = Close - (Range * 1.100 / 4)
    s2 = close_1d - (range_1d * 1.100 / 4)
    # S1 = Close - (Range * 1.100 / 2)
    s1 = close_1d - (range_1d * 1.100 / 2)
    # R1 = Close + (Range * 1.100 / 2)
    r1 = close_1d + (range_1d * 1.100 / 2)
    # R2 = Close + (Range * 1.100 / 4)
    r2 = close_1d + (range_1d * 1.100 / 4)
    # R3 = Close + (Range * 1.100 / 6)
    r3 = close_1d + (range_1d * 1.100 / 6)
    
    # Calculate ADX (14) on daily timeframe for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Plus Directional Movement (+DM)
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        
        # Minus Directional Movement (-DM)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        def Wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[1:period])  # Skip first NaN
                for i in range(period, len(data)):
                    if not np.isnan(data[i]):
                        result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        tr_smooth = Wilder_smoothing(tr, period)
        dm_plus_smooth = Wilder_smoothing(dm_plus, period)
        dm_minus_smooth = Wilder_smoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
        di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
        
        # DX
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        
        # ADX = smoothed DX
        adx = Wilder_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align 1-day indicators to 4h timeframe
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla levels, volume MA, and ADX
    start_idx = max(29, 19)  # 1-day data needs ~30 bars, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(s3_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume spike > 2x average
        volume_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price touches S3 level with volume spike and trending market
            if (price <= s3_1d_aligned[i] * 1.001 and  # Allow small tolerance
                volume_filter and trend_filter):
                signals[i] = size
                position = 1
            # Short: price touches R3 level with volume spike and trending market
            elif (price >= r3_1d_aligned[i] * 0.999 and  # Allow small tolerance
                  volume_filter and trend_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches S1 level (take profit) or reverses
            if price >= s1_1d_aligned[i] * 0.999:  # Reached S1 or better
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R1 level (take profit) or reverses
            if price <= r1_1d_aligned[i] * 1.001:  # Reached R1 or better
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_S3R3_Reversal_Volume_ADX"
timeframe = "4h"
leverage = 1.0