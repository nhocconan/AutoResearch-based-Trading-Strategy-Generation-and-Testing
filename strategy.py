#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Ichimoku cloud filter with 1d TK cross for entry timing and volume confirmation
# Long when price is above weekly Ichimoku cloud AND 1d Tenkan crosses above Kijun AND volume > 2.0 * avg_volume(20)
# Short when price is below weekly Ichimoku cloud AND 1d Tenkan crosses below Kijun AND volume > 2.0 * avg_volume(20)
# Exit when price crosses the weekly Kumo (cloud) boundary (Senkou Span A or B, whichever is closer to price)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Ichimoku cloud provides strong trend filter that works in both bull and bear markets
# 1d TK cross provides precise entry timing with lower false signals
# High volume threshold (2.0x) ensures only significant breakouts are traded, reducing fee drag

name = "6h_1wIchimoku_CloudFilter_1dTK_Cross_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Ichimoku cloud calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 completed weekly bars for Ichimoku (26*2)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    # Kumo (Cloud): area between Senkou Span A and Senkou Span B
    
    # Tenkan-sen (9-period)
    high_9_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_9_1w + low_9_1w) / 2.0
    
    # Kijun-sen (26-period)
    high_26_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_26_1w + low_26_1w) / 2.0
    
    # Senkou Span A
    senkou_span_a_1w = (tenkan_1w + kijun_1w) / 2.0
    
    # Senkou Span B (52-period)
    high_52_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1w = (high_52_1w + low_52_1w) / 2.0
    
    # Align weekly Ichimoku components to 6h timeframe (wait for completed weekly bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w)
    
    # Get 1d data ONCE before loop for TK cross calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:  # Need at least 26 completed daily bars for Kijun
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku Tenkan and Kijun for TK cross
    # Tenkan-sen (9-period) on 1d
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2.0
    
    # Kijun-sen (26-period) on 1d
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2.0
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries and position relative to cloud
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: price above weekly cloud AND 1d Tenkan crosses above Kijun AND volume spike
            if (close[i] > upper_cloud and 
                tenkan_1d_aligned[i] > kijun_1d_aligned[i] and 
                tenkan_1d_aligned[i-1] <= kijun_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly cloud AND 1d Tenkan crosses below Kijun AND volume spike
            elif (close[i] < lower_cloud and 
                  tenkan_1d_aligned[i] < kijun_1d_aligned[i] and 
                  tenkan_1d_aligned[i-1] >= kijun_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below the weekly cloud (lower boundary)
            if close[i] < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above the weekly cloud (upper boundary)
            if close[i] > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals