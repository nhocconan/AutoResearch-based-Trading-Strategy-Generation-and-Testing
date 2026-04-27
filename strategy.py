#!/usr/bin/env python3
"""
6h Camarilla Pivot R3/S3 Reversal with 1d Trend Filter
- Long at S3 bounce when 1d EMA34 is bullish
- Short at R3 rejection when 1d EMA34 is bearish
- Uses volume confirmation to avoid false breaks
- Target: 15-30 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(values, period):
    """Calculate EMA with proper handling of NaN"""
    ema = np.full_like(values, np.nan, dtype=np.float64)
    if len(values) < period:
        return ema
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + (range_val * 1.500)
    r3 = c + (range_val * 1.250)
    r2 = c + (range_val * 1.166)
    r1 = c + (range_val * 1.083)
    s1 = c - (range_val * 1.083)
    s2 = c - (range_val * 1.166)
    s3 = c - (range_val * 1.250)
    s4 = c - (range_val * 1.500)
    return r3, r2, r1, c, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    daily_close = df_1d['close'].values
    ema_34_1d = calculate_ema(daily_close, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels from previous day
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_prev = df_1d['close'].values
    
    # Initialize Camarilla arrays
    r3 = np.full_like(daily_close, np.nan)
    s3 = np.full_like(daily_close, np.nan)
    
    # Calculate Camarilla for each day (using previous day's data)
    for i in range(1, len(daily_close)):
        r3_i, _, _, _, _, _, s3_i, _ = calculate_camarilla(
            daily_high[i-1], daily_low[i-1], daily_close_prev[i-1]
        )
        r3[i] = r3_i
        s3[i] = s3_i
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate daily volume average for confirmation
    daily_volume = df_1d['volume'].values
    vol_ma_20 = np.full_like(daily_volume, np.nan, dtype=np.float64)
    for i in range(19, len(daily_volume)):
        vol_ma_20[i] = np.mean(daily_volume[i-19:i+1])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla (1d), EMA34 (1d), volume MA (1d)
    start_idx = 1  # Need previous day for Camarilla
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_aligned[i]
        
        # Current levels
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        
        # Volume filter: volume > 1.2x daily average
        vol_filter = vol_now > 1.2 * vol_ma
        
        if position == 0:
            # Long setup: price at S3 with bullish trend and volume
            if abs(price_now - s3_level) < 0.001 * s3_level and ema_trend > s3_level and vol_filter:
                signals[i] = size
                position = 1
            # Short setup: price at R3 with bearish trend and volume
            elif abs(price_now - r3_level) < 0.001 * r3_level and ema_trend < r3_level and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R3 or trend turns bearish
            if price_now >= r3_level or ema_trend < r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches S3 or trend turns bullish
            if price_now <= s3_level or ema_trend > s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3S3_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0