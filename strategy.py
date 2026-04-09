#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_filter_v1
# Hypothesis: 6h strategy using 1d Ichimoku cloud for trend direction and 6h Tenkan-Kijun cross for entries.
# Long: Price above 1d cloud (Senkou Span A/B) AND 6h Tenkan > Kijun (bullish momentum).
# Short: Price below 1d cloud AND 6h Tenkan < Kijun (bearish momentum).
# Exit: Tenkan-Kijun cross reverses OR price crosses cloud midpoint (Kumo Senkou).
# Uses 6h primary timeframe with 1d HTF for Ichimoku cloud filter.
# Designed for low trade frequency (~15-30/year) to avoid fee drag in ranging markets.
# Works in bull markets via trend-following and bear markets via short signals from cloud breaks.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h Ichimoku components (Tenkan, Kijun, Senkou Span A/B)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan = (high_s.rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
              low_s.rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun = (high_s.rolling(window=period_kijun, min_periods=period_kijun).max() + 
             low_s.rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_b = ((high_s.rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                 low_s.rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(26)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a.values, senkou_b.values)
    cloud_bottom = np.minimum(senkou_a.values, senkou_b.values)
    cloud_midpoint = (cloud_top + cloud_bottom) / 2
    
    # Get 1d data for HTF Ichimoku cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku components for cloud filter
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    close_1d_s = pd.Series(close_1d)
    
    # Tenkan-sen (1d)
    tenkan_1d = (high_1d_s.rolling(window=9, min_periods=9).max() + 
                 low_1d_s.rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (1d)
    kijun_1d = (high_1d_s.rolling(window=26, min_periods=26).max() + 
                low_1d_s.rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (1d)
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(26)
    
    # Senkou Span B (1d)
    senkou_b_1d = ((high_1d_s.rolling(window=52, min_periods=52).max() + 
                    low_1d_s.rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # 1d Cloud boundaries
    cloud_top_1d = np.maximum(senkou_a_1d.values, senkou_b_1d.values)
    cloud_bottom_1d = np.minimum(senkou_a_1d.values, senkou_b_1d.values)
    
    # Align 1d Ichimoku cloud to 6h
    cloud_top_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup (max lookback)
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(cloud_top_1d_aligned[i]) or np.isnan(cloud_bottom_1d_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # 6h Tenkan-Kijun cross signals
        tenkan_kijun_bullish = tenkan[i] > kijun[i]
        tenkan_kijun_bearish = tenkan[i] < kijun[i]
        
        # Price vs 1d cloud
        price_above_1d_cloud = close[i] > cloud_top_1d_aligned[i]
        price_below_1d_cloud = close[i] < cloud_bottom_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Tenkan-Kijun cross turns bearish OR price drops below cloud midpoint
            if (not tenkan_kijun_bullish) or (close[i] < cloud_midpoint[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan-Kijun cross turns bullish OR price rises above cloud midpoint
            if (not tenkan_kijun_bearish) or (close[i] > cloud_midpoint[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price above 1d cloud AND bullish Tenkan-Kijun cross
            if price_above_1d_cloud and tenkan_kijun_bullish:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below 1d cloud AND bearish Tenkan-Kijun cross
            elif price_below_1d_cloud and tenkan_kijun_bearish:
                position = -1
                signals[i] = -0.25
    
    return signals