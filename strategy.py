#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Weekly Trend Filter
Hypothesis: Ichimoku TK cross on 6h timeframe, filtered by weekly Kumo (cloud) direction,
captures trend continuation in both bull and bear markets while avoiding counter-trend signals.
Weekly cloud acts as a strong regime filter - price above weekly cloud = bullish bias,
price below = bearish bias. Targets 15-25 trades/year (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_weekly_trend"
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
    
    # Weekly data for trend filter (Kumo/cloud)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Weekly Kumo (cloud) components
    high_1w_9 = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_1w_9 = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_1w_9 + low_1w_9) / 2
    
    high_1w_26 = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_1w_26 = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_1w_26 + low_1w_26) / 2
    
    # Weekly Senkou Span A and B
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    high_1w_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_1w_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (high_1w_52 + low_1w_52) / 2
    
    # Align all indicators to avoid look-ahead
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need enough data for Ichimoku calculations
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly cloud boundaries and color
        weekly_leading_span_a = senkou_a_1w_aligned[i]
        weekly_leading_span_b = senkou_b_1w_aligned[i]
        weekly_kumo_top = max(weekly_leading_span_a, weekly_leading_span_b)
        weekly_kumo_bottom = min(weekly_leading_span_a, weekly_leading_span_b)
        weekly_kumo_bullish = weekly_leading_span_a > weekly_leading_span_b  # Green cloud
        
        if position == 1:  # Long position
            # Exit: TK cross down OR price closes below weekly cloud
            tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
            price_below_weekly_kumo = close[i] < weekly_kumo_bottom
            if tk_cross_down or price_below_weekly_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross up OR price closes above weekly cloud
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
            price_above_weekly_kumo = close[i] > weekly_kumo_top
            if tk_cross_up or price_above_weekly_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # TK cross signals
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
            tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
            
            # Price position relative to weekly cloud
            price_above_weekly_kumo = close[i] > weekly_kumo_top
            price_below_weekly_kumo = close[i] < weekly_kumo_bottom
            
            # Long: TK cross up + price above weekly cloud (bullish alignment)
            if tk_cross_up and price_above_weekly_kumo:
                position = 1
                signals[i] = 0.25
            # Short: TK cross down + price below weekly cloud (bearish alignment)
            elif tk_cross_down and price_below_weekly_kumo:
                position = -1
                signals[i] = -0.25
    
    return signals