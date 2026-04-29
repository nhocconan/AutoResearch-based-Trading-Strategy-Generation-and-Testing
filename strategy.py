#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK cross and 1d cloud filter
# Ichimoku provides objective support/resistance via Kumo (cloud)
# TK cross (Tenkan/Kijun) gives momentum signals with built-in confirmation
# 1d cloud filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-38/year) on 6h timeframe
# Works in both bull and bear markets by using cloud as dynamic support/resistance

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters: 9, 26, 52
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((high_52 + low_52) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Ichimoku components for entry signals
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_9_6h + low_9_6h) / 2
    
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_26_6h + low_26_6h) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Senkou Span A and B need to be shifted, but for simplicity we use current values
    # In practice, Senkou spans are plotted 26 periods ahead, but we use current for filter
    # The cloud is between Senkou A and Senkou B
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # Ichimoku warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries and price position relative to cloud
        senkou_a = senkou_a_1d_aligned[i]
        senkou_b = senkou_b_1d_aligned[i]
        
        # Cloud top is the higher of Senkou A and B
        cloud_top = max(senkou_a, senkou_b)
        # Cloud bottom is the lower of Senkou A and B
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Price above cloud: bullish, Price below cloud: bearish
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross signals
        tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price crosses below cloud OR TK cross turns bearish
            if not price_above_cloud or not tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above cloud OR TK cross turns bullish
            if not price_below_cloud or not tk_cross_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price above cloud + bullish TK cross
            if price_above_cloud and tk_cross_bullish:
                signals[i] = 0.25
                position = 1
            # Short entry: price below cloud + bearish TK cross
            elif price_below_cloud and tk_cross_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals