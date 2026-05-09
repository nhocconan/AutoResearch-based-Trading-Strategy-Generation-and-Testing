#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_1dTrend
# Hypothesis: Ichimoku TK cross with cloud filter on 1d timeframe for trend direction.
# In bullish 1d regime (price > cloud), buy when TK crosses above Kijun; in bearish regime (price < cloud), sell when TK crosses below Kijun.
# Uses 6h Tenkan/Kijun for entry timing, 1d Senkou Span for cloud filter. Designed for 50-150 trades over 4 years.
# Works in bull (follows cloud breakouts) and bear (fails when price breaks cloud opposite to trend).

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku components on 1d: Tenkan (9), Kijun (26), Senkou A/B (26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_9 = np.full_like(high_1d, np.nan)
    for i in range(8, len(high_1d)):
        tenkan_9[i] = (np.max(high_1d[i-8:i+1]) + np.min(low_1d[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_26 = np.full_like(high_1d, np.nan)
    for i in range(25, len(high_1d)):
        kijun_26[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = np.full_like(high_1d, np.nan)
    for i in range(len(tenkan_9)):
        if not np.isnan(tenkan_9[i]) and not np.isnan(kijun_26[i]):
            senkou_a[i] = (tenkan_9[i] + kijun_26[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    senkou_b = np.full_like(high_1d, np.nan)
    for i in range(51, len(high_1d)):
        senkou_b[i] = (np.max(high_1d[i-51:i+1]) + np.min(low_1d[i-51:i+1])) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    # Tenkan and Kijun are concurrent indicators (no look-ahead needed beyond bar close)
    tenkan_9_aligned = align_htf_to_ltf(prices, df_1d, tenkan_9)
    kijun_26_aligned = align_htf_to_ltf(prices, df_1d, kijun_26)
    # Senkou Span A/B are plotted 26 periods ahead, so we need to shift back 26 bars for current cloud
    # Since align_htf_to_ltf already waits for bar close, we use the values as-is for cloud calculation
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 6h Tenkan/Kijun for entry signals
    tenkan_6 = np.full_like(high, np.nan)
    kijun_6 = np.full_like(high, np.nan)
    for i in range(8, len(high)):
        tenkan_6[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    for i in range(25, len(high)):
        kijun_6[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # TK cross signals on 6h
    tk_cross_up = np.zeros_like(tenkan_6, dtype=bool)
    tk_cross_down = np.zeros_like(tenkan_6, dtype=bool)
    for i in range(1, len(tenkan_6)):
        if not np.isnan(tenkan_6[i]) and not np.isnan(kijun_6[i]) and not np.isnan(tenkan_6[i-1]) and not np.isnan(kijun_6[i-1]):
            tk_cross_up[i] = (tenkan_6[i-1] <= kijun_6[i-1]) and (tenkan_6[i] > kijun_6[i])
            tk_cross_down[i] = (tenkan_6[i-1] >= kijun_6[i-1]) and (tenkan_6[i] < kijun_6[i])
    
    # Cloud boundaries: Senkou Span A and B form the cloud
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Trend filter: price relative to cloud on 1d
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 1)  # Need Kijun period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(tenkan_6[i]) or np.isnan(kijun_6[i]) or \
           np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TK cross up AND price above cloud (bullish regime)
            if tk_cross_up[i] and price_above_cloud[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross down AND price below cloud (bearish regime)
            elif tk_cross_down[i] and price_below_cloud[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross down OR price breaks below cloud (trend change)
            if tk_cross_down[i] or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross up OR price breaks above cloud (trend change)
            if tk_cross_up[i] or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals