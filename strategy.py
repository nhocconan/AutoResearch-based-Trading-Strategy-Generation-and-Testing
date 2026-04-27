#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1wCloudFilter_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun cross provides timely entries, 
filtered by 1week cloud direction (trend) and 1d EMA50 trend alignment, with volume 
confirmation to avoid false breaks. Works in bull/bear via cloud trend filter. 
Target: 60-120 total trades over 4 years (15-30/year).
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
    
    # Get 1d data for EMA50 trend and Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for cloud (Senkou Span A/B) trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # --- 1d Ichimoku calculation (Tenkan, Kijun, Senkou Span A/B) ---
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen: 9-period
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # --- 1d EMA50 for additional trend filter ---
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- 1w Cloud trend: price above/both Senkou spans ---
    # For cloud trend, we need Senkou A/B from 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Senkou Span A/B on 1w
    # Tenkan 1w: 9-period
    high_9_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_9_1w + low_9_1w) / 2
    
    # Kijun 1w: 26-period
    high_26_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_26_1w + low_26_1w) / 2
    
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    
    # Senkou Span B 1w: 52-period
    high_52_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (high_52_1w + low_52_1w) / 2
    
    # Align all indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Cloud components: Senkou A/B from 1w (already plotted 26 periods ahead, so no extra delay needed for trend)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d Ichimoku (52), 1w cloud (52), volume avg (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(senkou_a_1w_aligned[i]) or 
            np.isnan(senkou_b_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        senkou_a_val = senkou_a_1w_aligned[i]
        senkou_b_val = senkou_b_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Determine cloud trend: price above/below both Senkou spans
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        if position == 0:
            # Tenkan-Kijun cross
            tk_cross_up = tenkan_val > kijun_val
            tk_cross_down = tenkan_val < kijun_val
            
            # Long conditions: TK cross up + price above cloud + 1d EMA50 uptrend + volume
            if (tk_cross_up and price_above_cloud and 
                close_val > ema_50_val and vol_conf):
                signals[i] = size
                position = 1
            
            # Short conditions: TK cross down + price below cloud + 1d EMA50 downtrend + volume
            elif (tk_cross_down and price_below_cloud and 
                  close_val < ema_50_val and vol_conf):
                signals[i] = -size
                position = -1
        
        elif position == 1:
            # Exit long: TK cross down OR price breaks below cloud bottom
            exit_condition = (tenkan_val < kijun_val) or (close_val < cloud_bottom)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        
        elif position == -1:
            # Exit short: TK cross up OR price breaks above cloud top
            exit_condition = (tenkan_val > kijun_val) or (close_val > cloud_top)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wCloudFilter_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0