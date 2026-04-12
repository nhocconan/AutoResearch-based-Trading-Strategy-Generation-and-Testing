#!/usr/bin/env python3
"""
6h_1d_ichimoku_cloud_with_volume
Hypothesis: Use Ichimoku Cloud from daily timeframe for trend direction and cloud as dynamic support/resistance.
Enter long when price is above cloud and Tenkan crosses above Kijun with volume confirmation.
Enter short when price is below cloud and Tenkan crosses below Kijun with volume confirmation.
Exit when price crosses back into the cloud or opposite cross occurs.
Works in bull/bear because cloud adapts to volatility and volume filters avoid false breakouts.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

name = "6h_1d_ichimoku_cloud_with_volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_high = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    tenkan_low = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_high = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    kijun_low = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b_high = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    senkou_b_low = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_b = (senkou_b_high + senkou_b_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals but could be used for confirmation
    
    # Shift Senkou spans forward by 26 periods (they are plotted ahead)
    senkou_a = np.roll(senkou_a, -kijun_period)
    senkou_b = np.roll(senkou_b, -kijun_period)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Wait for Ichimoku to be ready
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Check for Tenkan/Kijun cross
        tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
        kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
        tenkan_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_prev <= kijun_prev
        tenkan_cross_below = tenkan_aligned[i] < kijun_aligned[i] and tenkan_prev >= kijun_prev
        
        # Long entry: price above cloud, Tenkan crosses above Kijun, volume confirmation
        if (close[i] > cloud_top and tenkan_cross_above and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below cloud, Tenkan crosses below Kijun, volume confirmation
        elif (close[i] < cloud_bottom and tenkan_cross_below and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price crosses back into cloud or opposite cross
        elif position == 1 and (close[i] < cloud_top or tenkan_cross_below):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > cloud_bottom or tenkan_cross_above):
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