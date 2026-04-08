#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_volume_v1"
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
    
    # 1d data for Ichimoku calculation (weekly filter later)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud top and bottom (future cloud, shifted for look-ahead prevention)
    # Senkou spans are plotted 26 periods ahead, so we use current values for cloud
    cloud_top = np.maximum(span_a_6h, span_b_6h)
    cloud_bottom = np.minimum(span_a_6h, span_b_6h)
    
    # Weekly trend filter: price above/below weekly Ichimoku cloud
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Ichimoku (same parameters)
    wk_period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    wk_period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    wk_tenkan = (wk_period9_high + wk_period9_low) / 2.0
    
    wk_period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    wk_period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    wk_kijun = (wk_period26_high + wk_period26_low) / 2.0
    
    wk_senkou_a = (wk_tenkan + wk_kijun) / 2.0
    wk_period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    wk_period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    wk_senkou_b = (wk_period52_high + wk_period52_low) / 2.0
    
    wk_span_a = align_htf_to_ltf(prices, df_1w, wk_senkou_a)
    wk_span_b = align_htf_to_ltf(prices, df_1w, wk_senkou_b)
    wk_cloud_top = np.maximum(wk_span_a, wk_span_b)
    wk_cloud_bottom = np.minimum(wk_span_a, wk_span_b)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(wk_cloud_top[i]) or np.isnan(wk_cloud_bottom[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below cloud OR Tenkan-Kijun cross down
            if (close[i] < cloud_bottom[i] or 
                tenkan_6h[i] < kijun_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above cloud OR Tenkan-Kijun cross up
            if (close[i] > cloud_top[i] or 
                tenkan_6h[i] > kijun_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish: price above cloud, Tenkan > Kijun, weekly bullish
            bullish = (close[i] > cloud_top[i] and 
                      tenkan_6h[i] > kijun_6h[i] and
                      close[i] > wk_cloud_top[i])
            
            # Bearish: price below cloud, Tenkan < Kijun, weekly bearish
            bearish = (close[i] < cloud_bottom[i] and 
                      tenkan_6h[i] < kijun_6h[i] and
                      close[i] < wk_cloud_bottom[i])
            
            # Long entry: bullish + volume
            if bullish and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish + volume
            elif bearish and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals