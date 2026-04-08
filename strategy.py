#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Chikou Span (Lagging Span): close shifted back 26 periods
    chikou = np.roll(close_1d, 26)
    chikou[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Current price vs cloud (Senkou Span A and B)
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Senkou B calculation window
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(chikou_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below cloud bottom OR Tenkan-Kijun cross turns bearish
            if close[i] < cloud_bottom[i] or tenkan_6h[i] < kijun_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud top OR Tenkan-Kijun cross turns bullish
            if close[i] > cloud_top[i] or tenkan_6h[i] > kijun_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish: price above cloud, Tenkan > Kijun, Chikou above price from 26 periods ago
            bullish = (close[i] > cloud_top[i] and 
                      tenkan_6h[i] > kijun_6h[i] and 
                      chikou_6h[i] > close_1d[i] if not np.isnan(chikou_6h[i]) else False)
            
            # Bearish: price below cloud, Tenkan < Kijun, Chikou below price from 26 periods ago
            bearish = (close[i] < cloud_bottom[i] and 
                      tenkan_6h[i] < kijun_6h[i] and 
                      chikou_6h[i] < close_1d[i] if not np.isnan(chikou_6h[i]) else False)
            
            # Long entry: bullish Ichimoku + volume
            if bullish and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish Ichimoku + volume
            elif bearish and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals