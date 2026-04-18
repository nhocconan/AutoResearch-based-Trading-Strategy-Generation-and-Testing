#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Twist_With_WeeklyTrend
Hypothesis: Ichimoku cloud twist (Senkou A/B cross) signals trend change on daily timeframe.
Enter long when Tenkan > Kijun AND price above cloud AND weekly trend up (price > weekly SMA50).
Enter short when Tenkan < Kijun AND price below cloud AND weekly trend down (price < weekly SMA50).
Weekly trend filter avoids counter-trend trades in strong trends. Ichimoku provides dynamic support/resistance.
Works in bull/bear by following weekly trend. Low trade frequency due to multiple confluence requirements.
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
    volume = prices['volume'].values
    
    # Ichimoku parameters (standard)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    kijun_shift = 26
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over past 9 periods
    highest_high_9 = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    lowest_low_9 = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over past 26 periods
    highest_high_26 = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max()
    lowest_low_26 = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over past 52 periods shifted 26 ahead
    highest_high_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    lowest_low_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Current cloud: Senkou A/B from 26 periods ago (since they're plotted ahead)
    senkou_a_lagged = np.roll(senkou_a, kijun_shift)
    senkou_b_lagged = np.roll(senkou_b, kijun_shift)
    # First kijun_shift values are invalid due to roll
    senkou_a_lagged[:kijun_shift] = np.nan
    senkou_b_lagged[:kijun_shift] = np.nan
    
    # Weekly trend filter: price vs weekly SMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma_1w_50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_50)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after all indicators are valid
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, kijun_shift) + kijun_shift
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_lagged[i]) or np.isnan(senkou_b_lagged[i]) or
            np.isnan(sma_1w_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a_lagged[i]
        senkou_b_val = senkou_b_lagged[i]
        weekly_trend = sma_1w_50_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: Tenkan > Kijun (bullish cross) AND price above cloud AND weekly trend up
            if tenkan_val > kijun_val and price > cloud_top and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun (bearish cross) AND price below cloud AND weekly trend down
            elif tenkan_val < kijun_val and price < cloud_bottom and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: Tenkan < Kijun (bearish cross) OR price drops below cloud
            if tenkan_val < kijun_val or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: Tenkan > Kijun (bullish cross) OR price rises above cloud
            if tenkan_val > kijun_val or price > cloud_top:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Twist_With_WeeklyTrend"
timeframe = "6h"
leverage = 1.0