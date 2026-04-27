#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend_VolumeConfirm
Hypothesis: Uses 1d Ichimoku cloud (Senkou Span A/B) for trend direction and TK Cross (Tenkan/Kijun) for momentum.
Enter long when price > cloud (bullish) AND TK Cross bullish (Tenkan > Kijun) AND volume confirmation.
Enter short when price < cloud (bearish) AND TK Cross bearish (Tenkan < Kijun) AND volume confirmation.
Exit when TK Cross reverses or price crosses cloud middle. Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by following 1d Ichimoku trend while using TK Cross for timely entries.
Volume confirmation filter reduces false signals. Discrete position sizing (0.25) minimizes fee drag.
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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe (completed bars only)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: current volume > 1.8 * 30-period average (slightly looser for more trades)
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Senkou B (52) + TK Cross (26) + volume avg (30)
    start_idx = max(52, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        top_cloud = max(senkou_a_val, senkou_b_val)
        bottom_cloud = min(senkou_a_val, senkou_b_val)
        cloud_middle = (top_cloud + bottom_cloud) / 2
        
        if position == 0:
            # Look for entry: TK Cross with price/cloud relationship AND volume confirmation
            # Long: TK Cross bullish (Tenkan > Kijun) AND price above cloud AND volume confirmation
            long_condition = (tenkan_val > kijun_val) and (close_val > top_cloud) and vol_conf
            # Short: TK Cross bearish (Tenkan < Kijun) AND price below cloud AND volume confirmation
            short_condition = (tenkan_val < kijun_val) and (close_val < bottom_cloud) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when TK Cross turns bearish OR price crosses below cloud middle
            exit_condition = (tenkan_val <= kijun_val) or (close_val < cloud_middle)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when TK Cross turns bullish OR price crosses above cloud middle
            exit_condition = (tenkan_val >= kijun_val) or (close_val > cloud_middle)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0