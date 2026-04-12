#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
    # Uses 1d Senkou Span A/B for cloud filter: only take trades aligned with 1d cloud color
    # Tenkan/Kijun cross from 6h for entry timing with volume confirmation
    # Discrete sizing 0.25 to minimize fee churn. Target: 15-30 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.full(len(high_1d), np.nan)
    period9_low = np.full(len(low_1d), np.nan)
    for i in range(9, len(high_1d)):
        period9_high[i] = np.max(high_1d[i-9:i])
        period9_low[i] = np.min(low_1d[i-9:i])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.full(len(high_1d), np.nan)
    period26_low = np.full(len(low_1d), np.nan)
    for i in range(26, len(high_1d)):
        period26_high[i] = np.max(high_1d[i-26:i])
        period26_low[i] = np.min(low_1d[i-26:i])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = np.full(len(high_1d), np.nan)
    period52_low = np.full(len(low_1d), np.nan)
    for i in range(52, len(high_1d)):
        period52_high[i] = np.max(high_1d[i-52:i])
        period52_low[i] = np.min(low_1d[i-52:i])
    senkou_b = (period52_high + period52_low) / 2
    
    # Align 1d Ichimoku to 6h (cloud components)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 6h Tenkan and Kijun for entry signals
    period9_high_6h = np.full(n, np.nan)
    period9_low_6h = np.full(n, np.nan)
    for i in range(9, n):
        period9_high_6h[i] = np.max(high[i-9:i])
        period9_low_6h[i] = np.min(low[i-9:i])
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = np.full(n, np.nan)
    period26_low_6h = np.full(n, np.nan)
    for i in range(26, n):
        period26_high_6h[i] = np.max(high[i-26:i])
        period26_low_6h[i] = np.min(low[i-26:i])
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if data not ready
        if (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and price position relative to cloud
        green_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]  # bullish cloud
        red_cloud = senkou_a_aligned[i] < senkou_b_aligned[i]    # bearish cloud
        above_cloud = close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i])
        below_cloud = close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Tenkan/Kijun cross signals
        tenkan_above_kijun = tenkan_6h[i] > kijun_6h[i]
        tenkan_below_kijun = tenkan_6h[i] < kijun_6h[i]
        
        # Entry logic: aligned with cloud color and TK cross with volume
        long_entry = False
        short_entry = False
        
        # Long: bullish cloud + price above cloud + TK bullish cross + volume
        if green_cloud and above_cloud:
            long_entry = tenkan_above_kijun and volume_spike[i]
        # Short: bearish cloud + price below cloud + TK bearish cross + volume
        elif red_cloud and below_cloud:
            short_entry = tenkan_below_kijun and volume_spike[i]
        
        # Exit logic: opposite TK cross or cloud color change
        long_exit = tenkan_below_kijun or (green_cloud and not above_cloud) or (not green_cloud and not red_cloud)
        short_exit = tenkan_above_kijun or (red_cloud and not below_cloud) or (not green_cloud and not red_cloud)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_ichimoku_cloud_volume_v1"
timeframe = "6h"
leverage = 1.0