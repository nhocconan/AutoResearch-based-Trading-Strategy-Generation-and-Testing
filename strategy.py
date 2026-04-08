#!/usr/bin/env python3
"""
4h_12h1d_camarilla_pivot_v1
Hypothesis: Trade reversals at Camarilla pivot levels on 4h chart with 12h/1d trend filter and volume confirmation.
- Long at S3/S4 support in uptrend, short at R3/R4 resistance in downtrend
- Use 12h EMA25 and 1d EMA50 for trend alignment
- Require volume > 1.5x 20-bar average for confirmation
- Designed for low trade frequency (20-40/year) with clear invalidation levels
- Works in bull/bear via multi-timeframe trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h1d_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(values, period):
    """Calculate EMA with proper handling of NaN"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    ema = np.full_like(values, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    if range_val <= 0:
        return np.full_like(high, np.nan), np.full_like(high, np.nan), \
               np.full_like(high, np.nan), np.full_like(high, np.nan)
    close_val = close
    r4 = close_val + range_val * 1.1 / 2
    r3 = close_val + range_val * 1.1 / 4
    s3 = close_val - range_val * 1.1 / 4
    s4 = close_val - range_val * 1.1 / 2
    return r4, r3, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA25 for trend
    close_12h = df_12h['close'].values
    ema_25_12h = calculate_ema(close_12h, 25)
    
    # Calculate 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50_1d = calculate_ema(close_1d, 50)
    
    # Calculate Camarilla levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align indicators to 4h timeframe
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_25_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        trend_up_12h = price > ema_25_12h_aligned[i]
        trend_up_1d = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below S3 or trend turns down
            if price < s3_1d_aligned[i] or not (trend_up_12h and trend_up_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above R3 or trend turns up
            if price > r3_1d_aligned[i] or (trend_up_12h and trend_up_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price at S3/S4 with volume and uptrend
            if ((abs(price - s3_1d_aligned[i]) < 0.001 * price or abs(price - s4_1d_aligned[i]) < 0.001 * price) and
                vol_ratio > 1.5 and trend_up_12h and trend_up_1d):
                position = 1
                signals[i] = 0.25
            # Enter short: price at R3/R4 with volume and downtrend
            elif ((abs(price - r3_1d_aligned[i]) < 0.001 * price or abs(price - r4_1d_aligned[i]) < 0.001 * price) and
                  vol_ratio > 1.5 and not trend_up_12h and not trend_up_1d):
                position = -1
                signals[i] = -0.25
    
    return signals