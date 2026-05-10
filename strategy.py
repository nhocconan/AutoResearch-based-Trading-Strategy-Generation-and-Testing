#!/usr/bin/env python3
# 6H_1W_Ichimoku_Trend_Follow
# Hypothesis: Ichimoku cloud on weekly timeframe provides strong trend direction.
# Enter long when price crosses above Tenkan-sen and price is above Kumo (cloud) in weekly uptrend.
# Enter short when price crosses below Tenkan-sen and price is below Kumo in weekly downtrend.
# Uses weekly Ichimoku to capture major trend, reducing whipsaws in sideways markets.
# Works in bull/bear by following weekly trend. Target: 15-30 trades/year per symbol.

name = "6H_1W_Ichimoku_Trend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 weeks for Senkou B
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1w).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    low_9 = pd.Series(low_1w).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan_sen = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1w).rolling(window=kijun_period, min_periods=kijun_period).max()
    low_26 = pd.Series(low_1w).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun_sen = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1w).rolling(window=senkou_b_period, min_periods=senkou_b_period).max()
    low_52 = pd.Series(low_1w).rolling(window=senkou_b_period, min_periods=senkou_b_period).min()
    senkou_span_b = (high_52 + low_52) / 2.0
    
    # Chikou Span (Lagging Span): not used for signals, but we need values for alignment
    chikou_span = close_1w  # We'll align this but not use in signals
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for Ichimoku (need Senkou B)
    start_idx = senkou_b_period + kijun_period  # 52 + 26 = 78
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Kumo (cloud) boundaries - Senkou Span A and B shifted forward
        # For signal at time t, we use Senkou Span values that were plotted 26 periods ago
        # But since we're aligning, we need to check current price vs current cloud
        # The cloud ahead is already in senkou_span_a/b, but they represent future cloud
        # For simplicity, we use current Tenkan/Kijun vs current cloud (standard approach)
        
        # Determine if price is above or below cloud
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan-sen cross signals
        tenkan_cross_up = close[i] > tenkan_aligned[i] and close[i-1] <= tenkan_aligned[i-1]
        tenkan_cross_down = close[i] < tenkan_aligned[i] and close[i-1] >= tenkan_aligned[i-1]
        
        if position == 0:
            # Enter long: price above cloud + Tenkan crosses up
            if price_above_cloud and tenkan_cross_up:
                signals[i] = 0.25
                position = 1
            # Enter short: price below cloud + Tenkan crosses down
            elif price_below_cloud and tenkan_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below cloud or Tenkan crosses below Kijun
            if price_below_cloud or (close[i] < kijun_aligned[i] and close[i-1] >= kijun_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above cloud or Tenkan crosses above Kijun
            if price_above_cloud or (close[i] > kijun_aligned[i] and close[i-1] <= kijun_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals