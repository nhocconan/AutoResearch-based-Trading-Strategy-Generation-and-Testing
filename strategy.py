#!/usr/bin/env python3
"""
1h Camarilla Pivot Breakout with 4h EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as significant support/resistance derived from 4h data.
Breakouts above R3 or below S3 with volume confirmation and aligned 4h EMA34 trend capture swing moves.
Uses 4h timeframe for trend and pivots to reduce noise while maintaining alignment with 1h structure.
Designed for low trade frequency (15-37/year) with clear entry/exit rules to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot points"""
    if len(high) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    # Typical price for the period
    typical_price = (high + low + close) / 3
    # Range
    range_val = high - low
    
    # Camarilla levels
    pivot = typical_price
    r1 = close + (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    s1 = close - (range_val * 1.1 / 12)
    s2 = close - (range_val * 1.1 / 6)
    s3 = close - (range_val * 1.1 / 4)
    
    return (pivot, r1, r2, r3, s1, s2, s3)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and EMA34 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 100:
        return np.zeros(n)
    
    # Calculate Camarilla pivots on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Initialize arrays for Camarilla levels
    pivot_4h = np.full(len(close_4h), np.nan)
    r1_4h = np.full(len(close_4h), np.nan)
    r2_4h = np.full(len(close_4h), np.nan)
    r3_4h = np.full(len(close_4h), np.nan)
    s1_4h = np.full(len(close_4h), np.nan)
    s2_4h = np.full(len(close_4h), np.nan)
    s3_4h = np.full(len(close_4h), np.nan)
    
    # Calculate Camarilla for each 4h bar
    for i in range(len(df_4h)):
        if i < 1:  # Need at least one bar for calculation
            continue
        pivot, r1, r2, r3, s1, s2, s3 = calculate_camarilla(
            high_4h[i], low_4h[i], close_4h[i]
        )
        pivot_4h[i] = pivot
        r1_4h[i] = r1
        r2_4h[i] = r2
        r3_4h[i] = r3
        s1_4h[i] = s1
        s2_4h[i] = s2
        s3_4h[i] = s3
    
    # Align Camarilla components to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Calculate 34-period EMA on 4h close for trend
    ema_34_4h = calculate_ema(close_4h, 34)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 AND volume spike AND price > 4h EMA34 (uptrend)
            long_entry = (curr_close > r3_4h_aligned[i]) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below S3 AND volume spike AND price < 4h EMA34 (downtrend)
            short_entry = (curr_close < s3_4h_aligned[i]) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below S3 (support broken) OR price crosses below EMA (trend change)
            if (curr_close < s3_4h_aligned[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above R3 (resistance broken) OR price crosses above EMA (trend change)
            if (curr_close > r3_4h_aligned[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_Pivot_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0