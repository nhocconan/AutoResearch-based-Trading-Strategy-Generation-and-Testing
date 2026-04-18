#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1D
Hypothesis: Ichimoku Tenkan-Kijun cross with cloud filter from daily timeframe.
Tenkan (9) and Kijun (26) lines from 6h data, cloud (Senkou Span A/B) from daily.
In bull market: long when TK crosses above and price above cloud.
In bear market: short when TK crosses below and price below cloud.
Cloud acts as dynamic support/resistance, reducing whipsaws in sideways markets.
Target: 20-30 trades/year to stay within fee limits while capturing trend changes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Daily Ichimoku cloud for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Tenkan and Kijun
    high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_tenkan_1d + low_tenkan_1d) / 2.0
    
    high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_kijun_1d + low_kijun_1d) / 2.0
    
    # Daily Senkou Span A and B
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2.0
    high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_senkou_b_1d + low_senkou_b_1d) / 2.0
    
    # Align daily cloud to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 52  # Warmup for Senkou B
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud boundaries (future cloud, but we use current values for filtering)
        upper_cloud = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        price = close[i]
        
        if position == 0:
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
            
            # Long: bullish cross above cloud
            if tk_cross_up and price > upper_cloud:
                signals[i] = 0.25
                position = 1
            # Short: bearish cross below cloud
            elif tk_cross_down and price < lower_cloud:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: Tenkan crosses below Kijun OR price falls below cloud
            tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
            if tk_cross_down or price < lower_cloud:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: Tenkan crosses above Kijun OR price rises above cloud
            tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
            if tk_cross_up or price > upper_cloud:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1D"
timeframe = "6h"
leverage = 1.0