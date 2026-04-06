#!/usr/bin/env python3
"""
6h Ichimoku Cloud Strategy with Weekly Trend Filter
Hypothesis: Ichimoku Cloud provides dynamic support/resistance and trend signals. Using daily Tenkan/Kijun cross with weekly cloud color filter captures major trend continuations while avoiding counter-trend trades. Works in bull (buy when price above cloud, TK bullish, weekly bullish) and bear (sell when price below cloud, TK bearish, weekly bearish). Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan_sen[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun_sen[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        senkou_span_b[i] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Get weekly data for trend filter (cloud color)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly Ichimoku components (same parameters)
    weekly_tenkan = np.full(len(weekly_close), np.nan)
    weekly_kijun = np.full(len(weekly_close), np.nan)
    weekly_senkou_a = np.full(len(weekly_close), np.nan)
    weekly_senkou_b = np.full(len(weekly_close), np.nan)
    
    # Calculate weekly Tenkan-sen
    for i in range(tenkan_period - 1, len(weekly_close)):
        weekly_tenkan[i] = (np.max(weekly_high[i-tenkan_period+1:i+1]) + np.min(weekly_low[i-tenkan_period+1:i+1])) / 2
    
    # Calculate weekly Kijun-sen
    for i in range(kijun_period - 1, len(weekly_close)):
        weekly_kijun[i] = (np.max(weekly_high[i-kijun_period+1:i+1]) + np.min(weekly_low[i-kijun_period+1:i+1])) / 2
    
    # Calculate weekly Senkou Span A
    for i in range(kijun_period - 1, len(weekly_close)):
        if not np.isnan(weekly_tenkan[i]) and not np.isnan(weekly_kijun[i]):
            weekly_senkou_a[i] = (weekly_tenkan[i] + weekly_kijun[i]) / 2
    
    # Calculate weekly Senkou Span B
    for i in range(senkou_span_b_period - 1, len(weekly_close)):
        weekly_senkou_b[i] = (np.max(weekly_high[i-senkou_span_b_period+1:i+1]) + np.min(weekly_low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Weekly trend: bullish if Senkou A > Senkou B, bearish if Senkou A < Senkou B
    weekly_trend = np.where(weekly_senkou_a > weekly_senkou_b, 1, -1)
    
    # Align weekly trend to 6h timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need enough data for Ichimoku calculations)
    start = senkou_span_b_period  # 52 periods needed for Senkou B
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Determine cloud boundaries (shifted 26 periods ahead)
        # For signal at time i, we use Senkou Span values that were plotted 26 periods ago
        idx_a = i - kijun_period  # Senkou A plotted 26 periods ahead
        idx_b = i - kijun_period  # Senkou B plotted 26 periods ahead
        
        senkou_a_val = senkou_span_a[idx_a] if idx_a >= 0 and idx_a < n else np.nan
        senkou_b_val = senkou_span_b[idx_b] if idx_b >= 0 and idx_b < n else np.nan
        
        if np.isnan(senkou_a_val) or np.isnan(senkou_b_val):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Check exits
        if position == 1:  # long position
            # Exit: price falls below cloud OR TK cross turns bearish OR weekly trend turns bearish
            if (close[i] < cloud_bottom or
                tenkan_sen[i] < kijun_sen[i] or
                weekly_trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above cloud OR TK cross turns bullish OR weekly trend turns bullish
            if (close[i] > cloud_top or
                tenkan_sen[i] > kijun_sen[i] or
                weekly_trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries with minimum holding period
            if bars_since_entry >= 12:  # Minimum 12 bars (3 days) between entries
                # Bullish conditions: price above cloud, TK bullish, weekly bullish
                bullish = (close[i] > cloud_top and 
                          tenkan_sen[i] > kijun_sen[i] and 
                          weekly_trend_aligned[i] == 1)
                
                # Bearish conditions: price below cloud, TK bearish, weekly bearish
                bearish = (close[i] < cloud_bottom and 
                          tenkan_sen[i] < kijun_sen[i] and 
                          weekly_trend_aligned[i] == -1)
                
                if bullish:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bearish:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals