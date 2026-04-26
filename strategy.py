#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike
Hypothesis: Ichimoku cloud breakout on 6h with weekly trend filter (price > weekly cloud) and volume confirmation (>2.0x 20-bar MA). The Ichimoku system provides dynamic support/resistance via the cloud (Senkou Span A/B) and momentum via TK cross. Weekly trend filter ensures we only trade in the direction of the higher timeframe trend, reducing false breakouts. Volume spike confirms breakout strength. This combination works in both bull and bear markets by following the weekly trend while using Ichimoku for precise entries. Target: 50-150 total trades over 4 years.
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need enough for weekly Ichimoku calculations
        return np.zeros(n)
    
    # Calculate weekly Ichimoku cloud
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align weekly Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Calculate 6h Ichimoku components for entry signals
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    max_high_tenkan_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_low_tenkan_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (max_high_tenkan_6h + min_low_tenkan_6h) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    max_high_kijun_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    min_low_kijun_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (max_high_kijun_6h + min_low_kijun_6h) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (26 for Ichimoku, 20 for volume, 52 for weekly Senkou B)
    start_idx = max(26, 20, 52)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(tenkan_sen_6h[i]) or 
            np.isnan(kijun_sen_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        tenkan_6h = tenkan_sen_6h[i]
        kijun_6h = kijun_sen_6h[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine weekly trend: bullish if price above weekly cloud, bearish if below
        weekly_cloud_top = max(senkou_a_val, senkou_b_val)
        weekly_cloud_bottom = min(senkou_a_val, senkou_b_val)
        bullish_weekly = close_val > weekly_cloud_top
        bearish_weekly = close_val < weekly_cloud_bottom
        
        # TK cross: Tenkan-sen crossing above/below Kijun-sen
        tk_cross_up = tenkan_6h > kijun_6h
        tk_cross_down = tenkan_6h < kijun_6h
        
        # Price relative to cloud
        price_above_cloud = close_val > weekly_cloud_top
        price_below_cloud = close_val < weekly_cloud_bottom
        
        # Entry conditions: TK cross in direction of weekly trend with volume confirmation
        long_entry = tk_cross_up and bullish_weekly and vol_spike and price_above_cloud
        short_entry = tk_cross_down and bearish_weekly and vol_spike and price_below_cloud
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross turns down or price falls below cloud
            if tk_cross_down or close_val < weekly_cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit when TK cross turns up or price rises above cloud
            if tk_cross_up or close_val > weekly_cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0