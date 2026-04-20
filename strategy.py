#!/usr/bin/env python3
# 6h_12h_Ichimoku_Cloud_Trend
# Hypothesis: Use 12h Ichimoku cloud (Tenkan/Kijun) direction as trend filter on 6h.
# Enter long when price above cloud and Tenkan > Kijun, short when price below cloud and Tenkan < Kijun.
# Exit when price crosses back into cloud. Uses 12h for trend, 6h for entry/exit.
# Designed for 12-30 trades/year by requiring clear cloud breaks.
# Works in bull markets (trend following) and bear markets (avoids false signals via cloud filter).

name = "6h_12h_Ichimoku_Cloud_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    period_tenkan = 9
    period_kijun = 26
    period_senkou = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_high = pd.Series(high_12h).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    tenkan_low = pd.Series(low_12h).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_high = pd.Series(high_12h).rolling(window=period_kijun, min_periods=period_kijun).max()
    kijun_low = pd.Series(low_12h).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_high = pd.Series(high_12h).rolling(window=period_senkou, min_periods=period_senkou).max()
    senkou_low = pd.Series(low_12h).rolling(window=period_senkou, min_periods=period_senkou).min()
    senkou_b = ((senkou_high + senkou_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: price above cloud and Tenkan > Kijun (bullish)
            if (close[i] > cloud_top and tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud and Tenkan < Kijun (bearish)
            elif (close[i] < cloud_bottom and tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below cloud top
            if close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above cloud bottom
            if close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals