#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_Confirmation
Hypothesis: Fade at Camarilla R3/S3 levels with daily trend confirmation, using volume spikes to filter false breakouts.
Designed to work in both bull and bear markets by combining mean reversion at extreme intraday levels with higher timeframe trend alignment.
Uses Camarilla levels from daily pivots to identify overextended moves, with trend filter to avoid fading strong momentum.
Target: 50-150 total trades over 4 years (12-37/year) with disciplined risk management.
"""

name = "6h_Camarilla_R3_S3_Fade_1dTrend_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Pivot point
    pivot = (high + low + close) / 3
    # Range
    range_val = high - low
    # Camarilla levels
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r1, r2, r3, r4, s1, s2, s3, s4, pivot

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(prices).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    r1_1d, r2_1d, r3_1d, r4_1d, s1_1d, s2_1d, s3_1d, s4_1d, pivot_1d = calculate_camarilla_levels(
        high_1d[:-1], low_1d[:-1], close_1d[:-1]
    )
    # Add placeholder for first bar
    r1_1d = np.concatenate([[np.nan], r1_1d])
    r2_1d = np.concatenate([[np.nan], r2_1d])
    r3_1d = np.concatenate([[np.nan], r3_1d])
    r4_1d = np.concatenate([[np.nan], r4_1d])
    s1_1d = np.concatenate([[np.nan], s1_1d])
    s2_1d = np.concatenate([[np.nan], s2_1d])
    s3_1d = np.concatenate([[np.nan], s3_1d])
    s4_1d = np.concatenate([[np.nan], s4_1d])
    pivot_1d = np.concatenate([[np.nan], pivot_1d])
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = calculate_ema(close_1d, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long fade: price at or below S3 with uptrend on 1d AND volume spike
            if close[i] <= s3_1d[i] and ema_34_1d_aligned[i] > close[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short fade: price at or above R3 with downtrend on 1d AND volume spike
            elif close[i] >= r3_1d[i] and ema_34_1d_aligned[i] < close[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches midpoint OR trend breaks down
            if close[i] >= pivot_1d[i] or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price reaches midpoint OR trend breaks up
            if close[i] <= pivot_1d[i] or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals