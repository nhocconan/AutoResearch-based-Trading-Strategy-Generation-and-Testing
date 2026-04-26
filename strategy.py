#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h with cloud filter from 1d (Senkou Span A/B) provides high-probability entries aligned with daily trend. In bull markets, longs occur when price is above cloud and TK cross bullish; in bear markets, shorts when price below cloud and TK cross bearish. The cloud acts as dynamic support/resistance, reducing false signals. This strategy avoids overtrading by requiring both TK cross and cloud alignment, targeting 12-30 trades/year. Works in both bull and bear markets by using the cloud as trend filter and TK cross for timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # Calculate 1d Ichimoku cloud for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    max_high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_high_tenkan_1d + min_low_tenkan_1d) / 2.0
    
    # 1d Kijun-sen (26-period)
    max_high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_high_kijun_1d + min_low_kijun_1d) / 2.0
    
    # 1d Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2.0
    
    # 1d Senkou Span B (52-period)
    max_high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (max_high_senkou_b_1d + min_low_senkou_b_1d) / 2.0
    
    # Align 1d cloud components to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Determine cloud boundaries (always Senkou A as upper, Senkou B as lower in uptrend, but we take max/min)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # TK cross signals
    tk_cross_bullish = tenkan > kijun
    tk_cross_bearish = tenkan < kijun
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for all Ichimoku calculations (max 52 periods)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(tk_cross_bullish[i]) or np.isnan(tk_cross_bearish[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        
        if position == 0:
            # Flat - look for TK cross with cloud filter
            # Long: bullish TK cross AND price above cloud
            long_entry = tk_cross_bullish[i] and (close_val > cloud_top[i])
            # Short: bearish TK cross AND price below cloud
            short_entry = tk_cross_bearish[i] and (close_val < cloud_bottom[i])
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on bearish TK cross OR price drops below cloud
            exit_condition = tk_cross_bearish[i] or (close_val < cloud_top[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on bullish TK cross OR price rises above cloud
            exit_condition = tk_cross_bullish[i] or (close_val > cloud_bottom[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0