#!/usr/bin/env python3
"""
6h Ichimoku TK Cross + Cloud Filter from 1d + Volume Spike
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h with 1d cloud filter (price above/below cloud) 
captures strong trends while avoiding whipsaws. Volume spike confirms institutional 
participation. Works in bull/bear markets by trend-filtering via cloud.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculation (52) + volume MA
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # TK Cross signals with cloud filter
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud
            tk_cross_up = (tenkan_val > kijun_val) and (tenkan_6h[i-1] <= kijun_6h[i-1])
            price_above_cloud = curr_close > cloud_top[i]
            
            # Short: Tenkan crosses below Kijun AND price below cloud
            tk_cross_down = (tenkan_val < kijun_val) and (tenkan_6h[i-1] >= kijun_6h[i-1])
            price_below_cloud = curr_close < cloud_bottom[i]
            
            if tk_cross_up and price_above_cloud and volume_spike:
                signals[i] = 0.25
                position = 1
            elif tk_cross_down and price_below_cloud and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun OR price drops below cloud
            tk_cross_down = (tenkan_val < kijun_val) and (tenkan_6h[i-1] >= kijun_6h[i-1])
            price_below_cloud = curr_close < cloud_top[i]  # Exit if price drops below cloud top
            
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun OR price rises above cloud
            tk_cross_up = (tenkan_val > kijun_val) and (tenkan_6h[i-1] <= kijun_6h[i-1])
            price_above_cloud = curr_close > cloud_bottom[i]  # Exit if price rises above cloud bottom
            
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TKCross_CloudFilter_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0