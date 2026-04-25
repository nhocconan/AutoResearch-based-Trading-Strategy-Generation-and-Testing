#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrendFilter_v1
Hypothesis: Use 1d Camarilla pivot R3/S3 levels for breakout entries on 6h timeframe,
filtered by 1d EMA50 trend direction. Only take longs when price breaks above R3 in 1d uptrend,
or shorts when price breaks below S3 in 1d downtrend. Add volume confirmation (volume > 1.3 * 20-period MA).
Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or after 12 bars (3 days) to limit exposure.
Designed to capture sustained moves in both bull and bear markets with tight entries to minimize fee drag.
Target: 15-35 trades/year (60-140 total over 4 years).
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3, R4, S4)
    # Camarilla: based on previous day's OHLC
    df_1d_shift = df_1d.shift(1)
    high_1d = df_1d_shift['high'].values
    low_1d = df_1d_shift['low'].values
    close_1d = df_1d_shift['close'].values
    
    # Avoid look-ahead: use previous day's data only
    valid_idx = ~(np.isnan(high_1d) | np.isnan(low_1d) | np.isnan(close_1d))
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4.0)
    s3 = pivot - (range_1d * 1.1 / 4.0)
    r4 = pivot + (range_1d * 1.1 / 2.0)
    s4 = pivot - (range_1d * 1.1 / 2.0)
    
    # Align to 6h timeframe (1-day delay for pivot + alignment)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.3 * 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Determine 1d trend regime
        # Bull trend: price > EMA50
        # Bear trend: price < EMA50
        # No trend filter in range (avoid whipsaw)
        if close[i] > ema_50_1d_aligned[i]:
            trend = 'bull'
        elif close[i] < ema_50_1d_aligned[i]:
            trend = 'bear'
        else:
            trend = 'range'
        
        if position == 0:
            # Long entry: price breaks above R3 AND bull trend AND volume confirmation
            long_breakout = (close[i] > r3_aligned[i]) and (trend == 'bull') and volume_confirm[i]
            
            # Short entry: price breaks below S3 AND bear trend AND volume confirmation
            short_breakout = (close[i] < s3_aligned[i]) and (trend == 'bear') and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price reaches S3 (opposite level) OR trend turns bear OR max hold (12 bars = 3 days)
            if (close[i] <= s3_aligned[i]) or (trend == 'bear') or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price reaches R3 (opposite level) OR trend turns bull OR max hold (12 bars = 3 days)
            if (close[i] >= r3_aligned[i]) or (trend == 'bull') or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0