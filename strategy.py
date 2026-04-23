#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with Weekly Trend Filter and Volume Spike
- Ichimoku (Tenkan/Kijun) cross on 6h generates entry signals
- Weekly cloud (Senkou Span A/B) acts as trend filter: only long when price above cloud, short when below
- Volume confirmation (> 1.8x 24-period MA) reduces false signals
- Designed for 6h timeframe to capture swing moves with controlled frequency (target: 12-37 trades/year)
- Uses Ichimoku's built-in trend/momentum combination for robustness in both bull and bear markets
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
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Calculate weekly trend filter (cloud from 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Ichimoku components for cloud
    period_tenkan_w = 9
    max_high_tenkan_w = pd.Series(high_1w).rolling(window=period_tenkan_w, min_periods=period_tenkan_w).max().values
    min_low_tenkan_w = pd.Series(low_1w).rolling(window=period_tenkan_w, min_periods=period_tenkan_w).min().values
    tenkan_w = (max_high_tenkan_w + min_low_tenkan_w) / 2
    
    period_kijun_w = 26
    max_high_kijun_w = pd.Series(high_1w).rolling(window=period_kijun_w, min_periods=period_kijun_w).max().values
    min_low_kijun_w = pd.Series(low_1w).rolling(window=period_kijun_w, min_periods=period_kijun_w).min().values
    kijun_w = (max_high_kijun_w + min_low_kijun_w) / 2
    
    senkou_a_w = (tenkan_w + kijun_w) / 2
    
    period_senkou_b_w = 52
    max_high_senkou_b_w = pd.Series(high_1w).rolling(window=period_senkou_b_w, min_periods=period_senkou_b_w).max().values
    min_low_senkou_b_w = pd.Series(low_1w).rolling(window=period_senkou_b_w, min_periods=period_senkou_b_w).min().values
    senkou_b_w = (max_high_senkou_b_w + min_low_senkou_b_w) / 2
    
    # Weekly cloud boundaries (Senkou Span A/B)
    weekly_cloud_top = np.maximum(senkou_a_w, senkou_b_w)
    weekly_cloud_bottom = np.minimum(senkou_a_w, senkou_b_w)
    
    # Align weekly cloud to 6h timeframe
    weekly_cloud_top_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_top)
    weekly_cloud_bottom_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_bottom)
    
    # Volume confirmation: > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period_tenkan, period_kijun, period_senkou_b, 52, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(weekly_cloud_top_aligned[i]) or np.isnan(weekly_cloud_bottom_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above weekly cloud AND volume spike
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # bullish cross
                close[i] > weekly_cloud_top_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below weekly cloud AND volume spike
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # bearish cross
                  close[i] < weekly_cloud_bottom_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Tenkan/Kijun cross in opposite direction OR price crosses weekly cloud mid-point
            exit_signal = False
            weekly_cloud_mid = (weekly_cloud_top_aligned[i] + weekly_cloud_bottom_aligned[i]) / 2
            
            if position == 1:
                # Exit long on bearish cross OR price below weekly cloud
                if (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or close[i] < weekly_cloud_bottom_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on bullish cross OR price above weekly cloud
                if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or close[i] > weekly_cloud_top_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_WeeklyCloudTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0