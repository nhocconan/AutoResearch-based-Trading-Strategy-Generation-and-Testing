#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter
Hypothesis: 6s Ichimoku with daily cloud filter for trend alignment and momentum confirmation.
- Uses Ichimoku (Tenkan/Kijun/Senkou) on 6h for momentum and support/resistance
- Daily trend filter: price above/below daily Kumo cloud for long/short bias
- Entry: Tenkan crosses above Kijun + price above daily cloud (long)
         Tenkan crosses below Kijun + price below daily cloud (short)
- Exit: Reverse cross or price crosses cloud boundary
- Designed to capture momentum in trending markets while avoiding counter-trend noise
- Target: 20-35 trades/year on 6h (80-140 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Daily trend filter: price relative to Kumo cloud
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Daily Senkou Span A and B
    period_tenkan_d = 9
    period_kijun_d = 26
    period_senkou_b_d = 52
    
    max_high_tenkan_d = pd.Series(high_daily).rolling(window=period_tenkan_d, min_periods=period_tenkan_d).max().values
    min_low_tenkan_d = pd.Series(low_daily).rolling(window=period_tenkan_d, min_periods=period_tenkan_d).min().values
    tenkan_d = (max_high_tenkan_d + min_low_tenkan_d) / 2
    
    max_high_kijun_d = pd.Series(high_daily).rolling(window=period_kijun_d, min_periods=period_kijun_d).max().values
    min_low_kijun_d = pd.Series(low_daily).rolling(window=period_kijun_d, min_periods=period_kijun_d).min().values
    kijun_d = (max_high_kijun_d + min_low_kijun_d) / 2
    
    senkou_a_d = (tenkan_d + kijun_d) / 2
    
    max_high_senkou_b_d = pd.Series(high_daily).rolling(window=period_senkou_b_d, min_periods=period_senkou_b_d).max().values
    min_low_senkou_b_d = pd.Series(low_daily).rolling(window=period_senkou_b_d, min_periods=period_senkou_b_d).min().values
    senkou_b_d = (max_high_senkou_b_d + min_low_senkou_b_d) / 2
    
    # Align daily Ichimoku to 6h
    senkou_a_d_aligned = align_htf_to_ltf(prices, df_daily, senkou_a_d)
    senkou_b_d_aligned = align_htf_to_ltf(prices, df_daily, senkou_b_d)
    
    # Kumo cloud boundaries (Senkou A and B)
    upper_daily = np.maximum(senkou_a_d_aligned, senkou_b_d_aligned)
    lower_daily = np.minimum(senkou_a_d_aligned, senkou_b_d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for Ichimoku calculations
    start_idx = max(period_kijun, period_senkou_b) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_daily[i]) or np.isnan(lower_daily[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: Tenkan crosses above Kijun + price above daily cloud
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and close[i] > upper_daily[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Tenkan crosses below Kijun + price below daily cloud
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and close[i] < lower_daily[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun or price falls below cloud
            if (tenkan[i] < kijun[i] or close[i] < upper_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun or price rises above cloud
            if (tenkan[i] > kijun[i] or close[i] > lower_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Filter"
timeframe = "6h"
leverage = 1.0