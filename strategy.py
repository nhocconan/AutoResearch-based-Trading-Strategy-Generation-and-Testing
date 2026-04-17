#!/usr/bin/env python3
"""
6h_Advanced_Ichimoku_Pivot_Breakout
Advanced Ichimoku Cloud strategy with daily pivot confirmation on 6h timeframe.
Uses 1d Ichimoku cloud (Senkou Span A/B) for trend direction and Kumo twist detection,
combined with 1d pivot points (R1/S1) for breakout entries.
Designed for trending markets with pullback entries in both bull and bear regimes.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Ichimoku Cloud Components ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = np.full_like(high_1d, np.nan)
    period9_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 8:
            period9_high[i] = np.max(high_1d[i-8:i+1])
            period9_low[i] = np.min(low_1d[i-8:i+1])
        elif i > 0:
            period9_high[i] = np.max(high_1d[max(0, i-4):i+1])
            period9_low[i] = np.min(low_1d[max(0, i-4):i+1])
    
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = np.full_like(high_1d, np.nan)
    period26_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 25:
            period26_high[i] = np.max(high_1d[i-25:i+1])
            period26_low[i] = np.min(low_1d[i-25:i+1])
        elif i > 0:
            period26_high[i] = np.max(high_1d[max(0, i-12):i+1])
            period26_low[i] = np.min(low_1d[max(0, i-12):i+1])
    
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = np.full_like(high_1d, np.nan)
    period52_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 51:
            period52_high[i] = np.max(high_1d[i-51:i+1])
            period52_low[i] = np.min(low_1d[i-51:i+1])
        elif i > 0:
            period52_high[i] = np.max(high_1d[max(0, i-25):i+1])
            period52_low[i] = np.min(low_1d[max(0, i-25):i+1])
    
    senkou_b = ((period52_high + period52_low) / 2)
    
    # === 1d Pivot Points (Standard) ===
    # Pivot = (High + Low + Close)/3
    pivot = (high_1d + low_1d + close_1d) / 3
    # R1 = (2*Pivot) - Low
    r1 = (2 * pivot) - low_1d
    # S1 = (2*Pivot) - High
    s1 = (2 * pivot) - high_1d
    
    # === 6h Volume Confirmation (24-period average) ===
    vol_ma_24 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 23:
            vol_ma_24[i] = np.mean(volume[i-23:i+1])
        elif i > 0:
            vol_ma_24[i] = np.mean(volume[max(0, i-11):i+1])
    
    vol_confirm = volume > vol_ma_24 * 1.3  # volume spike: 1.3x average
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine cloud color and position
        # Green cloud: Senkou A > Senkou B (bullish)
        # Red cloud: Senkou A < Senkou B (bearish)
        green_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]
        red_cloud = senkou_a_aligned[i] < senkou_b_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Kumo twist detection (trend change signal)
        # Bullish twist: Senkou A crossing above Senkou B
        # Bearish twist: Senkou A crossing below Senkou B
        if i > warmup:
            prev_senkou_a = senkou_a_aligned[i-1]
            prev_senkou_b = senkou_b_aligned[i-1]
            bullish_twist = (prev_senkou_a <= prev_senkou_b) and (senkou_a_aligned[i] > senkou_b_aligned[i])
            bearish_twist = (prev_senkou_a >= prev_senkou_b) and (senkou_a_aligned[i] < senkou_b_aligned[i])
        else:
            bullish_twist = False
            bearish_twist = False
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long conditions:
            # 1. Bullish Kumo twist OR price above green cloud with bullish bias
            # 2. Price breaking above R1 resistance with volume
            # 3. Tenkan > Kijun (short-term momentum bullish)
            if ((bullish_twist or (green_cloud and price_above_cloud)) and
                (close[i] > r1_aligned[i]) and
                vol_confirm[i] and
                (tenkan_aligned[i] > kijun_aligned[i])):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short conditions:
            # 1. Bearish Kumo twist OR price below red cloud with bearish bias
            # 2. Price breaking below S1 support with volume
            # 3. Tenkan < Kijun (short-term momentum bearish)
            elif ((bearish_twist or (red_cloud and price_below_cloud)) and
                  (close[i] < s1_aligned[i]) and
                  vol_confirm[i] and
                  (tenkan_aligned[i] < kijun_aligned[i])):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Kumo turns red OR price falls below Tenkan-Kijun midpoint
            if (red_cloud or 
                close[i] < (tenkan_aligned[i] + kijun_aligned[i]) / 2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Kumo turns green OR price rises above Tenkan-Kijun midpoint
            if (green_cloud or 
                close[i] > (tenkan_aligned[i] + kijun_aligned[i]) / 2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Advanced_Ichimoku_Pivot_Breakout"
timeframe = "6h"
leverage = 1.0