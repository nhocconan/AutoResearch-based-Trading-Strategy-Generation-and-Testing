#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R1 with volume confirmation, short when breaks below S1. Use 1-day ADX > 25 to filter for trending markets only. Exit on opposite Camarilla level (S1 for longs, R1 for shorts). Uses tight entry conditions to limit trades (~25-35/year) and avoid fee drag. Works in both bull (breakouts) and bear (strong reversals at S1/R1).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # We'll calculate daily pivot points and then Camarilla levels
    # First get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    # First day has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.1 / 12)
    S1 = pivot - (range_val * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (wait for day close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get 1-day ADX for trend filter
    # Calculate TR, +DM, -DM
    tr1 = np.abs(np.diff(daily_high, prepend=daily_high[0]))
    tr2 = np.abs(np.diff(daily_low, prepend=daily_low[0]))
    tr3 = np.abs(daily_high - daily_low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    dm_plus = np.where((daily_high - np.roll(daily_high, 1)) > (np.roll(daily_low, 1) - daily_low), 
                       np.maximum(daily_high - np.roll(daily_high, 1), 0), 0)
    dm_minus = np.where((np.roll(daily_low, 1) - daily_low) > (daily_high - np.roll(daily_high, 1)), 
                        np.maximum(np.roll(daily_low, 1) - daily_low, 0), 0)
    # First values
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr14 = wilders_smoothing(tr, period)
    dm_plus_14 = wilders_smoothing(dm_plus, period)
    dm_minus_14 = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not volume_filter[i]:
            # If volume filter fails, maintain current position or flat
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume and ADX > 25
            if (close[i] > R1_aligned[i] and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume and ADX > 25
            elif (close[i] < S1_aligned[i] and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below S1 (opposite level)
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (opposite level)
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0