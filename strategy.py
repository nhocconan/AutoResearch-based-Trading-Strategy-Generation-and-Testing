#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend_v2
Hypothesis: Ichimoku TK cross with cloud filter on 6h, filtered by 1w trend (Tenkan/Kijun > Senkou Span B for long, < for short). 
Uses 1w trend to avoid counter-trend trades in strong trends, improving win rate in both bull and bear markets.
Target: 60-120 trades over 4 years (15-30/year) to stay within fee limits.
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
    
    # Ichimoku parameters (standard)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over past 9 periods
    highest_high_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    lowest_low_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (highest_high_tenkan + lowest_low_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 over past 26 periods
    highest_high_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max()
    lowest_low_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (highest_high_kijun + lowest_low_kijun) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 over past 52 periods
    highest_high_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    lowest_low_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = (highest_high_senkou_b + lowest_low_senkou_b) / 2
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    # Calculate 1w Ichimoku components for trend filter
    highest_high_1w_tenkan = pd.Series(high_1w).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    lowest_low_1w_tenkan = pd.Series(low_1w).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan_1w = (highest_high_1w_tenkan + lowest_low_1w_tenkan) / 2
    
    highest_high_1w_kijun = pd.Series(high_1w).rolling(window=kijun_period, min_periods=kijun_period).max()
    lowest_low_1w_kijun = pd.Series(low_1w).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun_1w = (highest_high_1w_kijun + lowest_low_1w_kijun) / 2
    
    highest_high_1w_senkou_b = pd.Series(high_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    lowest_low_1w_senkou_b = pd.Series(low_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b_1w = (highest_high_1w_senkou_b + lowest_low_1w_senkou_b) / 2
    
    # Align 1w trend filter to 6h timeframe
    # Trend condition: price > Senkou Span B (bullish) or price < Senkou Span B (bearish)
    # We'll use the average of Tenkan and Kijun for smoother trend signal
    avg_ik_1w = (tenkan_1w + kijun_1w) / 2
    trend_filter = align_htf_to_ltf(prices, df_1w, avg_ik_1w, additional_delay_bars=0)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w, additional_delay_bars=0)
    
    # Align 6h Ichimoku components
    tenkan_aligned = align_htf_to_ltf(prices, None, tenkan.values)  # Same timeframe, no alignment needed
    kijun_aligned = align_htf_to_ltf(prices, None, kijun.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, None, senkou_span_b.values)
    
    # For same timeframe alignment, we need to handle properly
    # Since we calculated on same data, we can use directly but need to handle NaN
    tenkan_vals = tenkan.values
    kijun_vals = kijun.values
    senkou_span_b_vals = senkou_span_b.values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period)  # 52
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_vals[i]) or 
            np.isnan(kijun_vals[i]) or
            np.isnan(senkou_span_b_vals[i]) or
            np.isnan(trend_filter[i]) or
            np.isnan(senkou_b_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan_val = tenkan_vals[i]
        kijun_val = kijun_vals[i]
        senkou_b_val = senkou_span_b_vals[i]
        trend_val = trend_filter[i]
        senkou_b_1w = senkou_b_1w_aligned[i]
        
        # TK cross signals
        tk_cross_up = tenkan_val > kijun_val and tenkan_vals[i-1] <= kijun_vals[i-1]
        tk_cross_down = tenkan_val < kijun_val and tenkan_vals[i-1] >= kijun_vals[i-1]
        
        # Cloud filter: price above/below cloud (using Senkou Span B as proxy for simplicity)
        price_above_cloud = price > senkou_b_val
        price_below_cloud = price < senkou_b_val
        
        # 1w trend filter: only trade in direction of higher timeframe trend
        # Bullish 1w trend: average IK > Senkou Span B
        # Bearish 1w trend: average IK < Senkou Span B
        bullish_1w_trend = trend_val > senkou_b_1w
        bearish_1w_trend = trend_val < senkou_b_1w
        
        if position == 0:
            # Long: TK cross up + price above cloud + bullish 1w trend
            if tk_cross_up and price_above_cloud and bullish_1w_trend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + bearish 1w trend
            elif tk_cross_down and price_below_cloud and bearish_1w_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TK cross down or price below cloud
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TK cross up or price above cloud
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend_v2"
timeframe = "6h"
leverage = 1.0