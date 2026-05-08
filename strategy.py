#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_TenkanKijun_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52)
    high_9 = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    high_26 = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    high_52 = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Price relative to cloud (senkou_a and senkou_b)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Tenkan crosses above Kijun, price above cloud, volume spike
            long_cond = (tenkan_6h[i] > kijun_6h[i] and 
                         price_above_cloud[i] and 
                         vol_spike[i])
            
            # Short entry: Tenkan crosses below Kijun, price below cloud, volume spike
            short_cond = (tenkan_6h[i] < kijun_6h[i] and 
                          price_below_cloud[i] and 
                          vol_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun
            if tenkan_6h[i] < kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun
            if tenkan_6h[i] > kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku Cloud with Tenkan-Kijun crossover on 6h timeframe using 1d Ichimoku components.
# Price above/below cloud filters false signals, volume spike confirms momentum.
# Works in bull markets (trend following via cloud) and bear markets (counter-trend via cloud rejection).
# Target: 15-30 trades/year to avoid fee drag while maintaining edge in both bull and bear markets.