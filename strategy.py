#!/usr/bin/env python3
# 1d_ichimoku_weekly_trend_v1
# Hypothesis: Uses Ichimoku cloud on 1d for trend direction and weekly cloud twist for confirmation.
# Long when: price above 1d cloud AND weekly Tenkan > Kijun (bullish twist).
# Short when: price below 1d cloud AND weekly Tenkan < Kijun (bearish twist).
# Exit when price crosses opposite cloud boundary.
# Designed for low frequency (<20 trades/year) to avoid fee drag, works in bull/bear via trend following.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ichimoku_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku on 1d: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen: (9-period high + 9-period low) / 2
    tenkan_sen = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan_sen[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    kijun_sen = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun_sen[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Senkou Span A: (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
    senkou_span_a = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + kijun_period  # displaced forward
            if idx < n:
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
    senkou_span_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        val = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
        idx = i + kijun_period  # displaced forward
        if idx < n:
            senkou_span_b[idx] = val
    
    # Get weekly data for cloud twist confirmation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Ichimoku components (same periods)
    tenkan_sen_1w = np.full(len(high_1w), np.nan)
    kijun_sen_1w = np.full(len(high_1w), np.nan)
    
    for i in range(tenkan_period - 1, len(high_1w)):
        tenkan_sen_1w[i] = (np.max(high_1w[i-tenkan_period+1:i+1]) + np.min(low_1w[i-tenkan_period+1:i+1])) / 2
    
    for i in range(kijun_period - 1, len(high_1w)):
        kijun_sen_1w[i] = (np.max(high_1w[i-kijun_period+1:i+1]) + np.min(low_1w[i-kijun_period+1:i+1])) / 2
    
    # Weekly bullish/bearish twist: Tenkan > Kijun (bullish) or Tenkan < Kijun (bearish)
    weekly_bullish_twist = tenkan_sen_1w > kijun_sen_1w
    weekly_bearish_twist = tenkan_sen_1w < kijun_sen_1w
    
    # Align weekly twist to daily
    weekly_bullish_twist_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish_twist.astype(float))
    weekly_bearish_twist_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish_twist.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = senkou_span_b_period + kijun_period - 1  # ensure all Ichimoku components available
    
    for i in range(start_idx, n):
        # Skip if cloud data not available
        if np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B form the cloud)
        upper_cloud = np.maximum(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        if position == 1:  # Long position
            # Exit: price drops below cloud
            if close[i] < lower_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud
            if close[i] > upper_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above cloud AND weekly bullish twist
            if (close[i] > upper_cloud and 
                weekly_bullish_twist_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short entry: price below cloud AND weekly bearish twist
            elif (close[i] < lower_cloud and 
                  weekly_bearish_twist_aligned[i] > 0.5):
                position = -1
                signals[i] = -0.25
    
    return signals