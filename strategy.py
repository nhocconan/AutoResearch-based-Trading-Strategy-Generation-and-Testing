# 2025-06-23 | 6h_Ichimoku_Cloud_Trend_v1
# Hypothesis: Use Ichimoku Cloud from 1d timeframe for trend direction and 6h for entry timing.
# In bull markets: price above cloud + Tenkan > Kijun = bullish trend.
# In bear markets: price below cloud + Tenkan < Kijun = bearish trend.
# The Ichimoku cloud provides dynamic support/resistance and adapts to volatility.
# Cloud acts as a filter: only trade in direction of cloud color (green=bull, red=bear).
# Entry on 6h when Tenkan crosses Kijun with confirmation from cloud and price action.
# Designed for low trade frequency (15-35/year) to minimize fee drag in both bull and bear markets.

name = "6h_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.full_like(high_1d, np.nan)
    period9_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 8:
            start_idx = i - 8
            period9_high[i] = np.max(high_1d[start_idx:i+1])
            period9_low[i] = np.min(low_1d[start_idx:i+1])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.full_like(high_1d, np.nan)
    period26_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 25:
            start_idx = i - 25
            period26_high[i] = np.max(high_1d[start_idx:i+1])
            period26_low[i] = np.min(low_1d[start_idx:i+1])
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.full_like(high_1d, np.nan)
    period52_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 51:
            start_idx = i - 51
            period52_high[i] = np.max(high_1d[start_idx:i+1])
            period52_low[i] = np.min(low_1d[start_idx:i+1])
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 51  # Need Senkou B (52-period) to be ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud color and position
        # Green cloud: senkou_a > senkou_b (bullish)
        # Red cloud: senkou_a < senkou_b (bearish)
        cloud_green = senkou_a_6h[i] > senkou_b_6h[i]
        cloud_red = senkou_a_6h[i] < senkou_b_6h[i]
        
        # Price position relative to cloud
        price_above_cloud = close[i] > max(senkou_a_6h[i], senkou_b_6h[i])
        price_below_cloud = close[i] < min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Enter long: bullish alignment
            # Price above green cloud AND Tenkan > Kijun
            if cloud_green and price_above_cloud and tenkan_6h[i] > kijun_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment
            # Price below red cloud AND Tenkan < Kijun
            elif cloud_red and price_below_cloud and tenkan_6h[i] < kijun_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below cloud OR Tenkan crosses below Kijun
            if price_below_cloud or tenkan_6h[i] < kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above cloud OR Tenkan crosses above Kijun
            if price_above_cloud or tenkan_6h[i] > kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals