#!/usr/bin/env python3
"""
6h_ichimoku_cloud_1d_trend_v1
Hypothesis: Use Ichimoku cloud from daily timeframe as trend filter, with Tenkan-Kijun cross on 6h for entry.
In bull markets (price above weekly cloud), take long signals from TK cross; in bear markets (price below weekly cloud), take short signals.
Weekly cloud acts as regime filter to avoid counter-trend trades. Designed for ~15-25 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku components (tenkan, kijun, senkou span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(26)
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Get 1w data for regime filter (weekly cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku cloud
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Tenkan-sen and Kijun-sen
    tenkan_sen_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    
    # Weekly Senkou Span A and B
    senkou_span_a_1w = ((tenkan_sen_1w + kijun_sen_1w) / 2).shift(26)
    senkou_span_b_1w = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align weekly cloud to 6h
    senkou_span_a_1w_6h = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w.values)
    senkou_span_b_1w_6h = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w.values)
    
    # Determine cloud boundaries (top and bottom of cloud)
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    daily_cloud_top = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    daily_cloud_bottom = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    weekly_cloud_top = np.maximum(senkou_span_a_1w_6h, senkou_span_b_1w_6h)
    weekly_cloud_bottom = np.minimum(senkou_span_a_1w_6h, senkou_span_b_1w_6h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any data is not available
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(daily_cloud_top[i]) or np.isnan(daily_cloud_bottom[i]) or
            np.isnan(weekly_cloud_top[i]) or np.isnan(weekly_cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime using weekly cloud
        price_above_weekly_cloud = close[i] > weekly_cloud_top[i]
        price_below_weekly_cloud = close[i] < weekly_cloud_bottom[i]
        
        # TK cross signals on 6h timeframe
        tk_cross_bull = tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
        tk_cross_bear = tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
        
        if position == 1:  # Long position
            # Exit: price crosses below daily cloud base or bearish TK cross in bear regime
            if close[i] < daily_cloud_bottom[i] or (tk_cross_bear and price_below_weekly_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above daily cloud top or bullish TK cross in bull regime
            if close[i] > daily_cloud_top[i] or (tk_cross_bull and price_above_weekly_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # In bull regime (price above weekly cloud): look for bullish TK cross
            if price_above_weekly_cloud and tk_cross_bull:
                # Additional filter: price should be above daily cloud for stronger bullish signal
                if close[i] > daily_cloud_top[i]:
                    position = 1
                    signals[i] = 0.25
            # In bear regime (price below weekly cloud): look for bearish TK cross
            elif price_below_weekly_cloud and tk_cross_bear:
                # Additional filter: price should be below daily cloud for stronger bearish signal
                if close[i] < daily_cloud_bottom[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals