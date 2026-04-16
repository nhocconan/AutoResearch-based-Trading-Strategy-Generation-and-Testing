#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter
Hypothesis: Ichimoku Tenkan/Kijun cross with daily cloud filter provides high-probability 
trend-following entries. Tenkan (9-period) crossing above/below Kijun (26-period) signals 
momentum shifts. Only take trades when price is above/below the daily Kumo (cloud) to 
align with higher-timeframe trend. This avoids counter-trend whipsaws and works in both 
bull/bear markets by following the dominant trend. Uses volume confirmation to filter 
false breaks. Target: 50-150 trades over 4 years.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for Ichimoku cloud) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Ichimoku calculations on 6h timeframe ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): close shifted 22 periods behind (not used for signals)
    
    # === Daily Ichimoku Cloud (for trend filter) ===
    # Calculate cloud on daily timeframe
    high_1d_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_1d_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_1d_9 + low_1d_9) / 2
    
    high_1d_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_1d_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_1d_26 + low_1d_26) / 2
    
    high_1d_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_1d_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = ((high_1d_52 + low_1d_52) / 2)
    
    # The cloud boundaries: Senkou Span A and B
    # Actual cloud is plotted 26 periods ahead, so we use current values for Senkou A/B
    # but the cloud itself represents future support/resistance
    # For filtering, we check if price is above/both Senkou lines (bullish cloud) 
    # or below/both (bearish cloud)
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # === Volume confirmation on 6h ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_6h / vol_ma_20
    
    # === Align all HTF data to 6h timeframe ===
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Ichimoku calculations (max 52 periods)
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or 
            np.isnan(cloud_bottom_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        cloud_top = cloud_top_aligned[i]
        cloud_bottom = cloud_bottom_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun OR price falls below cloud
            if tenkan_val < kijun_val or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun OR price rises above cloud
            if tenkan_val > kijun_val or price > cloud_top:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish: Tenkan crosses above Kijun with volume, price above cloud
            if (tenkan_val > kijun_val and 
                price > cloud_top and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
                continue
            # Bearish: Tenkan crosses below Kijun with volume, price below cloud
            elif (tenkan_val < kijun_val and 
                  price < cloud_bottom and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter"
timeframe = "6h"
leverage = 1.0