#!/usr/bin/env python3
"""
6h_1w_Ichimoku_Cloud_Trend_v1
Hypothesis: 6h Ichimoku Tenkan/Kijun cross with weekly cloud filter and volume confirmation. Uses weekly Ichimoku cloud (Senkou A/B) to determine trend direction, with Tenkan-Kijun cross as entry signal and Kijun as dynamic stop. Works in bull/bear via cloud color filter and avoids whipsaws in sideways markets by requiring cloud thickness > 0.5% of price.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Ichimoku_Cloud_Trend_v1"
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
    
    # === WEEKLY ICHIMOKU ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # need at least 52 weeks for calculations
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    def ichimoku_components(high_arr, low_arr, close_arr):
        # Tenkan-sen (Conversion Line): (9-period high + low)/2
        period9_high = np.full_like(high_arr, np.nan)
        period9_low = np.full_like(low_arr, np.nan)
        for i in range(len(high_arr)):
            if i >= 8:
                period9_high[i] = np.max(high_arr[i-8:i+1])
                period9_low[i] = np.min(low_arr[i-8:i+1])
        tenkan = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + low)/2
        period26_high = np.full_like(high_arr, np.nan)
        period26_low = np.full_like(low_arr, np.nan)
        for i in range(len(high_arr)):
            if i >= 25:
                period26_high[i] = np.max(high_arr[i-25:i+1])
                period26_low[i] = np.min(low_arr[i-25:i+1])
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
        period52_high = np.full_like(high_arr, np.nan)
        period52_low = np.full_like(low_arr, np.nan)
        for i in range(len(high_arr)):
            if i >= 51:
                period52_high[i] = np.max(high_arr[i-51:i+1])
                period52_low[i] = np.min(low_arr[i-51:i+1])
        senkou_b = (period52_high + period52_low) / 2
        
        return tenkan, kijun, senkou_a, senkou_b
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = ichimoku_components(high_1w, low_1w, close_1w)
    
    # Cloud top and bottom (Senkou A/B)
    cloud_top = np.maximum(senkou_a_1w, senkou_b_1w)
    cloud_bottom = np.minimum(senkou_a_1w, senkou_b_1w)
    cloud_thickness = cloud_top - cloud_bottom
    
    # Align weekly Ichimoku to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1w, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1w, cloud_bottom)
    cloud_thickness_aligned = align_htf_to_ltf(prices, df_1w, cloud_thickness)
    
    # Volume average (24-period for 6h = ~6 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 24:
            vol_sum -= volume[i-24]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or 
            np.isnan(cloud_thickness_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Cloud filter: price above/below cloud and cloud thick enough
        price_above_cloud = close[i] > cloud_top_aligned[i]
        price_below_cloud = close[i] < cloud_bottom_aligned[i]
        cloud_filter = cloud_thickness_aligned[i] > (close[i] * 0.005)  # >0.5% thickness
        
        # Tenkan-Kijun cross signals
        tenkan_cross_above = (tenkan_1w_aligned[i] > kijun_1w_aligned[i]) and (i == 100 or tenkan_1w_aligned[i-1] <= kijun_1w_aligned[i-1])
        tenkan_cross_below = (tenkan_1w_aligned[i] < kijun_1w_aligned[i]) and (i == 100 or tenkan_1w_aligned[i-1] >= kijun_1w_aligned[i-1])
        
        # Entry conditions
        long_entry = price_above_cloud and tenkan_cross_above and vol_confirm and cloud_filter
        short_entry = price_below_cloud and tenkan_cross_below and vol_confirm and cloud_filter
        
        # Exit when price crosses Kijun (dynamic stop)
        exit_long = close[i] < kijun_1w_aligned[i]
        exit_short = close[i] > kijun_1w_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals