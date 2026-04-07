#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud with 1d Filter
# Hypothesis: Ichimoku provides trend direction, momentum, and support/resistance.
# Tenkan/Kijun cross gives entry signals, cloud acts as dynamic support/resistance.
# 1d cloud filter ensures alignment with higher timeframe trend.
# Works in bull via bullish crosses above cloud, in bear via bearish crosses below cloud.
# Target: 15-30 trades/year to minimize fee drag.
name = "6h_ichimoku_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Get 1d Ichimoku cloud for filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2
    
    # Align 1d cloud components to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        # Determine 1d cloud boundaries
        upper_cloud_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Price position relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # 1d trend filter: price relative to 1d cloud
        price_above_1d_cloud = close[i] > upper_cloud_1d
        price_below_1d_cloud = close[i] < lower_cloud_1d
        
        if position == 1:  # Long position
            # Exit: price crosses below Tenkan or re-enters cloud
            if close[i] < tenkan[i] or (close[i] < upper_cloud and close[i] > lower_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above Tenkan or re-enters cloud
            if close[i] > tenkan[i] or (close[i] < upper_cloud and close[i] > lower_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Tenkan crosses above Kijun AND price above cloud AND 1d bullish
            if (tenkan[i] > kijun[i] and 
                price_above_cloud and 
                price_above_1d_cloud):
                position = 1
                signals[i] = 0.25
            # Enter short: Tenkan crosses below Kijun AND price below cloud AND 1d bearish
            elif (tenkan[i] < kijun[i] and 
                  price_below_cloud and 
                  price_below_1d_cloud):
                position = -1
                signals[i] = -0.25
    
    return signals