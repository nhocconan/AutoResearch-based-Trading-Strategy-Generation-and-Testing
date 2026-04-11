#!/usr/bin/env python3
# 6h_1d_ichimoku_kumo_v1
# Strategy: 6h Ichimoku Cloud with 1d cloud filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Ichimoku Cloud acts as dynamic support/resistance. Price above/below cloud indicates trend. 1d cloud filter ensures alignment with higher timeframe trend, reducing false signals. Works in both bull/bear by trading with the trend on pullbacks to cloud edges.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_kumo_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    highest_senkou = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    lowest_senkou = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = (highest_senkou + lowest_senkou) / 2
    
    # Shift Senkou spans forward by 26 periods
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # 1d Ichimoku Cloud for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    highest_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    lowest_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_1d = (highest_tenkan_1d + lowest_tenkan_1d) / 2
    
    # 1d Kijun-sen (26-period)
    highest_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    lowest_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_1d = (highest_kijun_1d + lowest_kijun_1d) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B (52-period)
    highest_senkou_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    lowest_senkou_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b_1d = (highest_senkou_1d + lowest_senkou_1d) / 2
    
    # Align 1d cloud components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # Calculate 1d cloud boundaries (shifted)
    senkou_a_1d_shifted = np.roll(senkou_a_1d_aligned, 26)
    senkou_b_1d_shifted = np.roll(senkou_b_1d_aligned, 26)
    senkou_a_1d_shifted[:26] = np.nan
    senkou_b_1d_shifted[:26] = np.nan
    
    # 1d Cloud top and bottom
    cloud_top_1d = np.maximum(senkou_a_1d_shifted, senkou_b_1d_shifted)
    cloud_bottom_1d = np.minimum(senkou_a_1d_shifted, senkou_b_1d_shifted)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 6h cloud boundaries
        cloud_top = max(senkou_a_shifted[i], senkou_b_shifted[i])
        cloud_bottom = min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        # Trend filter: price relative to 1d cloud
        price_above_1d_cloud = close[i] > cloud_top_1d[i]
        price_below_1d_cloud = close[i] < cloud_bottom_1d[i]
        
        # Entry conditions
        # Long: Price above 6h cloud AND above 1d cloud (uptrend alignment)
        if price_above_1d_cloud and close[i] > cloud_top and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price below 6h cloud AND below 1d cloud (downtrend alignment)
        elif price_below_1d_cloud and close[i] < cloud_bottom and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses opposite cloud boundary or 1d cloud flip
        elif position == 1 and (close[i] < cloud_bottom or not price_above_1d_cloud):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > cloud_top or not price_below_1d_cloud):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals