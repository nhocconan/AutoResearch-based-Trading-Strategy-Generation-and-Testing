#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_Volume_Trend
Hypothesis: Ichimoku cloud from daily timeframe acts as dynamic support/resistance.
Tenkan-Kijun cross above/below cloud with volume confirmation captures momentum shifts.
Works in bull/bear by following cloud direction and institutional volume.
Target: 15-25 trades/year (60-100 total over 4 years) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Senkou B
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = 52  # Warmup for Ichimoku (need 52 days)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        vol_ok = volume_filter[i]
        
        # Determine cloud relationship (price above/below cloud)
        price_above_cloud = price > cloud_top_val
        price_below_cloud = price < cloud_bottom_val
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud with volume
            if (tenkan_val > kijun_val and price_above_cloud and vol_ok):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Tenkan crosses below Kijun AND price below cloud with volume
            elif (tenkan_val < kijun_val and price_below_cloud and vol_ok):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 3 bars (1.5 days)
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: price returns to cloud or Tenkan crosses below Kijun
                if price < cloud_bottom_val or tenkan_val < kijun_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 3 bars (1.5 days)
            if bars_since_entry < 3:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: price returns to cloud or Tenkan crosses above Kijun
                if price > cloud_top_val or tenkan_val > kijun_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0