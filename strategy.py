#!/usr/bin/env python3
"""
6h_ichimoku_cloud_1d_trend_v1
Hypothesis: On 6h timeframe, use Ichimoku cloud (TK cross) for entry signals with 1d trend filter (price above/below cloud) to capture trends in both bull and bear markets. Cloud acts as dynamic support/resistance. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components on 6h (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2)
    
    # Get daily data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily for trend filter
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Daily Tenkan and Kijun
    dh9 = pd.Series(daily_high).rolling(window=9, min_periods=9).max()
    dl9 = pd.Series(daily_low).rolling(window=9, min_periods=9).min()
    d_tenkan = ((dh9 + dl9) / 2).values
    
    dh26 = pd.Series(daily_high).rolling(window=26, min_periods=26).max()
    dl26 = pd.Series(daily_low).rolling(window=26, min_periods=26).min()
    d_kijun = ((dh26 + dl26) / 2).values
    
    # Daily Senkou Span A and B
    d_senkou_a = ((d_tenkan + d_kijun) / 2)
    dh52 = pd.Series(daily_high).rolling(window=52, min_periods=52).max()
    dl52 = pd.Series(daily_low).rolling(window=52, min_periods=52).min()
    d_senkou_b = ((dh52 + dl52) / 2)
    
    # Align daily Ichimoku components to 6h (shifted by 1 day)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, d_tenkan)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, d_kijun)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_a)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # 1d Cloud boundaries
        cloud_top_1d = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK cross signals
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # 1d trend filter: price relative to 1d cloud
        price_above_1d_cloud = close[i] > cloud_top_1d
        price_below_1d_cloud = close[i] < cloud_bottom_1d
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on TK cross down
            if tk_cross_down:
                exit_long = True
            # Exit when price falls below cloud
            elif close[i] < cloud_top:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on TK cross up
            if tk_cross_up:
                exit_short = True
            # Exit when price rises above cloud
            elif close[i] > cloud_bottom:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TK cross up, price above cloud, 1d trend up (price above 1d cloud)
            long_entry = tk_cross_up and price_above_cloud and price_above_1d_cloud
            
            # Short entry: TK cross down, price below cloud, 1d trend down (price below 1d cloud)
            short_entry = tk_cross_down and price_below_cloud and price_below_1d_cloud
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals