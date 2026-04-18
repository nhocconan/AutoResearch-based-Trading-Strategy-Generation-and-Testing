#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian breakout with weekly pivot direction filter and volume confirmation.
In trending markets (price above/below weekly pivot), trade Donchian(20) breakouts.
In ranging markets (price within weekly pivot range), avoid trades to prevent whipsaw.
Volume confirms institutional participation at breakouts.
Designed for 15-25 trades/year to minimize fee shock while capturing clean trends.
Works in bull markets (buy breakouts above weekly pivot) and bear markets (sell breakdowns below weekly pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channel upper and lower bands."""
    upper = np.full_like(high, np.nan, dtype=float)
    lower = np.full_like(high, np.nan, dtype=float)
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-(period-1):i+1])
        lower[i] = np.min(low[i-(period-1):i+1])
    return upper, lower

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot levels (standard 5-point)."""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, s1, r2, s2, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot filter
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot levels
    pivot_w, r1_w, s1_w, r2_w, s2_w, r3_w, s3_w = calculate_weekly_pivot(high_w, low_w, close_w)
    
    # Align weekly pivot to 6h
    pivot_w_6h = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_6h = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_6h = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_6h = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_6h = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Calculate Donchian(20) on 6h
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Calculate volume moving average (24-period = 4 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need Donchian calculation and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_w_6h[i]) or np.isnan(r2_w_6h[i]) or 
            np.isnan(s2_w_6h[i]) or np.isnan(vol_ma[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i])):
            signals[i] = 0.0
            continue
        
        # Weekly pivot context: trending if price outside S2/R2, ranging if inside
        price_vs_pivot = close[i]
        in_range = (price_vs_pivot >= s2_w_6h[i]) and (price_vs_pivot <= r2_w_6h[i])
        above_pivot = price_vs_pivot > pivot_w_6h[i]
        below_pivot = price_vs_pivot < pivot_w_6h[i]
        
        # Volume confirmation: current volume > 1.5 * 24-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade in trending markets (outside weekly S2/R2)
            if not in_range:
                # Long breakout above Donchian upper with volume
                if close[i] > donch_upper[i] and vol_confirmed and above_pivot:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below Donchian lower with volume
                elif close[i] < donch_lower[i] and vol_confirmed and below_pivot:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: return to weekly pivot or breakdown below Donchian lower
            if close[i] <= pivot_w_6h[i] or close[i] < donch_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to weekly pivot or breakout above Donchian upper
            if close[i] >= pivot_w_6h[i] or close[i] > donch_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0