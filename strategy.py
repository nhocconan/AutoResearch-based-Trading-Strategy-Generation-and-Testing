#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1d
Hypothesis: Use Ichimoku Tenkan-Kijun cross on 6h with 1d cloud as trend filter. 
Tenkan (9-period) and Kijun (26-period) cross signals momentum shifts. 
Only take longs when price above 1d cloud (bullish bias) and shorts when below (bearish bias). 
Add volume > 1.5x 24-period average for confirmation. 
Ichimoku works in trends by capturing momentum and in ranges by avoiding false crosses via cloud filter.
Targets 15-30 trades/year (~60-120 total) via strict cloud filter + volume confirmation.
Works in bull/bear by following higher timeframe trend bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for cloud (Senkou Span A/B) and Tenkan/Kijun
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    tenkan_1d = np.full_like(close_1d, np.nan)
    kijun_1d = np.full_like(close_1d, np.nan)
    senkou_span_a_1d = np.full_like(close_1d, np.nan)
    senkou_span_b_1d = np.full_like(close_1d, np.nan)
    
    if len(high_1d) >= senkou_span_b_period:
        # Tenkan-sen
        for i in range(tenkan_period - 1, len(high_1d)):
            tenkan_1d[i] = (np.max(high_1d[i - tenkan_period + 1:i + 1]) + np.min(low_1d[i - tenkan_period + 1:i + 1])) / 2
        
        # Kijun-sen
        for i in range(kijun_period - 1, len(high_1d)):
            kijun_1d[i] = (np.max(high_1d[i - kijun_period + 1:i + 1]) + np.min(low_1d[i - kijun_period + 1:i + 1])) / 2
        
        # Senkou Span B
        for i in range(senkou_span_b_period - 1, len(high_1d)):
            senkou_span_b_1d[i] = (np.max(high_1d[i - senkou_span_b_period + 1:i + 1]) + np.min(low_1d[i - senkou_span_b_period + 1:i + 1])) / 2
        
        # Senkou Span A: (Tenkan + Kijun)/2, plotted 26 periods ahead
        # We'll calculate it properly by shifting later
        for i in range(kijun_period - 1, len(high_1d)):
            senkou_span_a_1d[i] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    # Align 1d Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # 6h Tenkan and Kijun for entry signal
    tenkan_6h = np.full_like(close, np.nan)
    kijun_6h = np.full_like(close, np.nan)
    
    if len(high) >= kijun_period:
        for i in range(tenkan_period - 1, len(high)):
            tenkan_6h[i] = (np.max(high[i - tenkan_period + 1:i + 1]) + np.min(low[i - tenkan_period + 1:i + 1])) / 2
        
        for i in range(kijun_period - 1, len(high)):
            kijun_6h[i] = (np.max(high[i - kijun_period + 1:i + 1]) + np.min(low[i - kijun_period + 1:i + 1])) / 2
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, vol_period) + 26  # +26 for Senkou Span A lookahead
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Cloud boundaries (Senkou Span A/B plotted 26 periods ahead, so we use current values)
        # For cloud filter, we compare current price to the cloud plotted 26 periods ago
        # But since we aligned, we need to check if we have the plotted values
        # Simpler: use current Senkou Span values as cloud boundaries for filtering
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun + price above cloud + volume
            if i > 0 and not np.isnan(tenkan_6h[i-1]) and not np.isnan(kijun_6h[i-1]):
                if tenkan_6h[i-1] <= kijun_6h[i-1] and tenkan_6h[i] > kijun_6h[i]:
                    if close[i] > upper_cloud and vol_confirm:
                        signals[i] = 0.25
                        position = 1
            # Short: Tenkan crosses below Kijun + price below cloud + volume
            elif i > 0 and not np.isnan(tenkan_6h[i-1]) and not np.isnan(kijun_6h[i-1]):
                if tenkan_6h[i-1] >= kijun_6h[i-1] and tenkan_6h[i] < kijun_6h[i]:
                    if close[i] < lower_cloud and vol_confirm:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun or price falls below cloud
            if (i > 0 and not np.isnan(tenkan_6h[i-1]) and not np.isnan(kijun_6h[i-1]) and 
                tenkan_6h[i-1] > kijun_6h[i-1] and tenkan_6h[i] < kijun_6h[i]) or close[i] < lower_cloud:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun or price rises above cloud
            if (i > 0 and not np.isnan(tenkan_6h[i-1]) and not np.isnan(kijun_6h[i-1]) and 
                tenkan_6h[i-1] < kijun_6h[i-1] and tenkan_6h[i] > kijun_6h[i]) or close[i] > upper_cloud:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0