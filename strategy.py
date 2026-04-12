#!/usr/bin/env python3
# 6h_12h_ichimoku_trend_follow
# Hypothesis: 6-hour Ichimoku trend following with 12-hour cloud filter
# Uses Ichimoku TK cross on 6h for entry/exit, filtered by 12h cloud (price above/below cloud)
# Works in bull/bear by requiring price to be on correct side of higher timeframe cloud
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag

name = "6h_12h_ichimoku_trend_follow"
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
    
    # Get 12h data for Ichimoku and cloud filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Determine cloud boundaries (Senkou Span A/B)
    # Cloud top = max(Senkou A, Senkou B)
    # Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after Ichimoku warmup
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: TK cross bullish AND price above cloud
        if (tenkan_aligned[i] > kijun_aligned[i] and 
            close[i] > cloud_top[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: TK cross bearish AND price below cloud
        elif (tenkan_aligned[i] < kijun_aligned[i] and 
              close[i] < cloud_bottom[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: TK cross reverses OR price crosses cloud
        elif position == 1 and (tenkan_aligned[i] < kijun_aligned[i] or close[i] < cloud_bottom[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_aligned[i] > kijun_aligned[i] or close[i] > cloud_top[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals