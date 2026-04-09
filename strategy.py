#!/usr/bin/env python3
# 6h_ichimoku_1d_cloud_filter_v2
# Hypothesis: 6h strategy using 1d Ichimoku cloud for trend direction with 6h Tenkan-Kijun cross for entry timing.
# Long: Price above 1d Ichimoku cloud AND 6h Tenkan crosses above Kijun.
# Short: Price below 1d Ichimoku cloud AND 6h Tenkan crosses below Kijun.
# Exit: Opposite Tenkan-Kijun cross OR price re-enters the cloud.
# Uses 6h primary timeframe with 1d HTF for Ichimoku cloud filter.
# Designed for low trade frequency (~15-30/year) to minimize fee drag while capturing major trends.
# Works in bull markets via cloud trend following and bear markets via shorting below cloud.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_cloud_filter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    tenkan = (high_s.rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
              low_s.rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan = tenkan.values
    
    # 6h Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun = (high_s.rolling(window=period_kijun, min_periods=period_kijun).max() + 
             low_s.rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun = kijun.values
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_1d = tenkan_1d.values
    
    # 1d Kijun-sen (Base Line): (26-period high + low)/2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_1d = kijun_1d.values
    
    # 1d Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan_1d + kijun_1d) / 2)
    # Shift forward 26 periods (will be aligned later with align_htf_to_ltf)
    
    # 1d Senkou Span B (Leading Span B): (52-period high + low)/2 plotted 26 periods ahead
    senkou_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    senkou_b = senkou_b.values
    # Shift forward 26 periods (will be aligned later with align_htf_to_ltf)
    
    # Align 1d Ichimoku components to 6h (cloud edges are plotted ahead, so no extra delay needed)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate cloud boundaries (after alignment)
    senkou_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    senkou_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup for 6h indicators
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or i < 2 or
            np.isnan(senkou_top[i]) or np.isnan(senkou_bottom[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # 6h Tenkan-Kijun cross signals
        tenkan_prev = tenkan[i-1]
        kijun_prev = kijun[i-1]
        tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan[i] > kijun[i])
        tk_cross_down = (tenkan_prev >= kijun_prev) and (tenkan[i] < kijun[i])
        
        # Price relative to 1d cloud
        price_above_cloud = close[i] > senkou_top[i]
        price_below_cloud = close[i] < senkou_bottom[i]
        
        if position == 1:  # Long position
            # Exit: Tenkan-Kijun cross down OR price re-enters cloud
            if tk_cross_down or not price_above_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan-Kijun cross up OR price re-enters cloud
            if tk_cross_up or not price_below_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price above cloud AND Tenkan-Kijun cross up
            if price_above_cloud and tk_cross_up:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud AND Tenkan-Kijun cross down
            elif price_below_cloud and tk_cross_down:
                position = -1
                signals[i] = -0.25
    
    return signals