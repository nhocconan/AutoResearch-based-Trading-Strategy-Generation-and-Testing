#!/usr/bin/env python3
"""
6h Weekly Pivot Reversal with Volume Confirmation.
Long when price breaks above weekly R1 with volume confirmation in bullish regime (price > weekly pivot).
Short when price breaks below weekly S1 with volume confirmation in bearish regime (price < weekly pivot).
Exit when price returns to weekly pivot level.
Designed to generate 50-150 total trades over 4 years with reversal edge in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivots(high, low, close):
    """Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot, r1, s1 = calculate_weekly_pivots(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Volume filter: volume > 1.3x average to avoid false signals
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly data + volume MA
    start_idx = max(19, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current weekly levels
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Volume filter
        vol_filter = vol_now > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R1 in bullish regime (price > pivot) + volume
            if price_now > r1_val and price_now > pivot_val and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 in bearish regime (price < pivot) + volume
            elif price_now < s1_val and price_now < pivot_val and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to or below pivot
            if price_now <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to or above pivot
            if price_now >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_Reversal_Volume"
timeframe = "6h"
leverage = 1.0