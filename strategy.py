#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud with Tenkan/Kijun cross and Senkou Span filters.
- Uses daily Senkou Span A/B for cloud (support/resistance)
- Tenkan (9) and Kijun (26) cross for momentum signals
- Filters trades to occur only when price is above/below cloud appropriately
- Works in bull/bear markets by using cloud as dynamic support/resistance
- Target: 12-37 trades/year per symbol (50-150 total over 4 years) to minimize fee drag
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # need at least 26*2 for Ichimoku
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all Ichimoku components
    start_idx = 52  # max lookback for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        
        # Determine cloud boundaries (Senkou Span A and B form the cloud)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Ichimoku signal: Tenkan/Kijun cross
        # Bullish cross: Tenkan crosses above Kijun
        # Bearish cross: Tenkan crosses below Kijun
        if i > start_idx:
            tenkan_prev = tenkan_6h[i-1]
            kijun_prev = kijun_6h[i-1]
            bullish_cross = tenkan_prev <= kijun_prev and tenkan_val > kijun_val
            bearish_cross = tenkan_prev >= kijun_prev and tenkan_val < kijun_val
        else:
            bullish_cross = False
            bearish_cross = False
        
        # Entry conditions
        if position == 0:
            # Long: bullish cross AND price above cloud
            if bullish_cross and close[i] > upper_cloud:
                signals[i] = size
                position = 1
            # Short: bearish cross AND price below cloud
            elif bearish_cross and close[i] < lower_cloud:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish cross OR price drops below cloud
            if bearish_cross or close[i] < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish cross OR price rises above cloud
            if bullish_cross or close[i] > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Cross_Filter"
timeframe = "6h"
leverage = 1.0