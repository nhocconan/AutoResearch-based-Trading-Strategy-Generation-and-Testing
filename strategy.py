#!/usr/bin/env python3
"""
6h_ichimoku_cloud_1d_trend_v1
Hypothesis: On 6h timeframe, use Ichimoku system with cloud filter from 1d timeframe to identify high-probability trend continuation entries. Long when Tenkan > Kijun and price above cloud (from 1d), short when Tenkan < Kijun and price below cloud. Uses 1d cloud color (Senkou A > B for bullish, < for bearish) as trend filter. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while capturing trend momentum in both bull and bear markets via multi-timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_ata, align_htf_to_ltf

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
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    if len(high) < period_tenkan:
        return np.zeros(n)
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    if len(high) < period_kijun:
        return np.zeros(n)
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    if len(high) < period_senkou_b:
        return np.zeros(n)
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2.0
    
    # Calculate 1d Ichimoku for trend filter (cloud color and position)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    max_high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_high_9_1d + min_low_9_1d) / 2.0
    
    # 1d Kijun-sen (26-period)
    max_high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_high_26_1d + min_low_26_1d) / 2.0
    
    # 1d Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2.0
    
    # 1d Senkou Span B (52-period)
    max_high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (max_high_52_1d + min_low_52_1d) / 2.0
    
    # 1d Cloud color: Senkou A > Senkou B = bullish cloud, < = bearish cloud
    cloud_bullish_1d = senkou_a_1d > senkou_b_1d
    
    # Align 1d Ichimoku components to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    cloud_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bullish_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(senkou_a_1d_aligned[i]) or 
            np.isnan(senkou_b_1d_aligned[i]) or np.isnan(cloud_bullish_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun OR price falls below cloud
            if tenkan[i] < kijun[i] or (close[i] < senkou_a[i] and close[i] < senkou_b[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun OR price rises above cloud
            if tenkan[i] > kijun[i] or (close[i] > senkou_a[i] and close[i] > senkou_b[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine cloud boundaries (future cloud, already displaced 26 periods)
            upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
            lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
            
            # Long: Tenkan > Kijun AND price above cloud AND 1d cloud bullish
            if tenkan[i] > kijun[i] and close[i] > upper_cloud and cloud_bullish_1d_aligned[i] > 0.5:
                position = 1
                signals[i] = 0.25
            # Short: Tenkan < Kijun AND price below cloud AND 1d cloud bearish
            elif tenkan[i] < kijun[i] and close[i] < lower_cloud and cloud_bullish_1d_aligned[i] < 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals