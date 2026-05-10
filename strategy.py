#!/usr/bin/env python3
"""
6h_Ichimoku_Kijun_Sen_Rebound
Hypothesis: Price rebounds from Ichimoku Kijun Sen (26-period) on 6h timeframe when aligned with 1d Kumo (cloud) trend.
In trending markets, price respects Kijun Sen as dynamic support/resistance. Cloud color from 1d filter ensures
trading in direction of higher timeframe trend. Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_Ichimoku_Kijun_Sen_Rebound"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d Ichimoku components for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan Sen (9-period): (HH9 + LL9)/2
    tenkan_9 = np.full(len(high_1d), np.nan)
    kijun_26 = np.full(len(high_1d), np.nan)
    senkou_span_a = np.full(len(high_1d), np.nan)
    senkou_span_b = np.full(len(high_1d), np.nan)
    
    if len(high_1d) >= 26:
        # Calculate Tenkan Sen (9)
        for i in range(8, len(high_1d)):
            hh9 = np.max(high_1d[i-8:i+1])
            ll9 = np.min(low_1d[i-8:i+1])
            tenkan_9[i] = (hh9 + ll9) / 2
        
        # Calculate Kijun Sen (26)
        for i in range(25, len(high_1d)):
            hh26 = np.max(high_1d[i-25:i+1])
            ll26 = np.min(low_1d[i-25:i+1])
            kijun_26[i] = (hh26 + ll26) / 2
        
        # Senkou Span A: (Tenkan + Kijun)/2 plotted 26 periods ahead
        for i in range(len(tenkan_9)):
            if not np.isnan(tenkan_9[i]) and not np.isnan(kijun_26[i]):
                idx = i + 26
                if idx < len(senkou_span_a):
                    senkou_span_a[idx] = (tenkan_9[i] + kijun_26[i]) / 2
        
        # Senkou Span B: 52-period HL/2 plotted 26 periods ahead
        if len(high_1d) >= 52:
            for i in range(51, len(high_1d)):
                hh52 = np.max(high_1d[i-51:i+1])
                ll52 = np.min(low_1d[i-51:i+1])
                idx = i + 26
                if idx < len(senkou_span_b):
                    senkou_span_b[idx] = (hh52 + ll52) / 2
    
    # Align 1d Ichimoku to 6h
    kijun_26_aligned = align_htf_to_ltf(prices, df_1d, kijun_26)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 6h Ichimoku Kijun Sen (26-period) for entry signal
    kijun_26_6h = np.full(n, np.nan)
    if n >= 26:
        for i in range(25, n):
            hh26 = np.max(high[i-25:i+1])
            ll26 = np.min(low[i-25:i+1])
            kijun_26_6h[i] = (hh26 + ll26) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kijun_26_aligned[i]) or np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or np.isnan(kijun_26_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d cloud color and position
        senkou_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        senkou_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_green = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]  # bullish cloud
        price_above_cloud = close[i] > senkou_top
        price_below_cloud = close[i] < senkou_bottom
        
        if position == 0:
            # Long: Price rebounds from 6h Kijun Sen in bullish cloud
            if (cloud_green and close[i] <= kijun_26_6h[i] * 1.005 and  # near Kijun Sen
                close[i] > kijun_26_6h[i] and  # above Kijun Sen
                price_above_cloud):  # above cloud (strong uptrend)
                signals[i] = 0.25
                position = 1
            # Short: Price rebounds from 6h Kijun Sen in bearish cloud
            elif (not cloud_green and close[i] >= kijun_26_6h[i] * 0.995 and  # near Kijun Sen
                  close[i] < kijun_26_6h[i] and  # below Kijun Sen
                  price_below_cloud):  # below cloud (strong downtrend)
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below Kijun Sen or cloud turns bearish
            if close[i] < kijun_26_6h[i] * 0.995 or not cloud_green:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above Kijun Sen or cloud turns bullish
            if close[i] > kijun_26_6h[i] * 1.005 or cloud_green:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals