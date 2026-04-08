#!/usr/bin/env python3
# 6h_ichimoku_cloud_band_trend_follow_v1
# Hypothesis: Ichimoku cloud (from 1d) provides directional bias; Tenkan-Kijun cross on 6h triggers entries when price is above/below cloud. Works in bull/bear because cloud acts as dynamic support/resistance and trend filter. Uses Tenkan/Kijun cross for entry timing with cloud as filter to avoid whipsaws.

name = "6h_ichimoku_cloud_band_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku components (calculated on 1d data)
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    # Leading Span A: (Conversion Line + Base Line) / 2
    # Leading Span B: (52-period high + 52-period low) / 2
    # The cloud is between Leading Span A and B
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Lagging Span
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): 9-period
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): 26-period
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): 52-period
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe (with proper shift for forward projection)
    # Senkou Span A and B are plotted 26 periods ahead
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 6h Tenkan-Kijun cross for entry signals
    # Tenkan-sen (9-period) on 6h data
    max_high_tenkan_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_low_tenkan_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (max_high_tenkan_6h + min_low_tenkan_6h) / 2
    
    # Kijun-sen (26-period) on 6h data
    max_high_kijun_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    min_low_kijun_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (max_high_kijun_6h + min_low_kijun_6h) / 2
    
    # Start from sufficient lookback
    start_idx = max(26, 52) + 1
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Leading Span A and B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine if price is above or below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan-Kijun cross on 6h
        tenkan_kijun_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tenkan_kijun_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 1:  # Long position
            # Exit if price falls below cloud or Tenkan crosses below Kijun
            if price_below_cloud or tenkan_kijun_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price rises above cloud or Tenkan crosses above Kijun
            if price_above_cloud or tenkan_kijun_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above cloud AND Tenkan crosses above Kijun
            if price_above_cloud and tenkan_kijun_cross_up:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud AND Tenkan crosses below Kijun
            elif price_below_cloud and tenkan_kijun_cross_down:
                position = -1
                signals[i] = -0.25
    
    return signals