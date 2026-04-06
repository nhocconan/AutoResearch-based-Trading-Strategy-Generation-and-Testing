#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Filter
Hypothesis: Ichimoku provides strong trend signals (TK cross) and dynamic support/resistance (cloud). Filtering by 1d price relative to cloud ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws. Works in bull (buy when price above cloud + TK cross up) and bear (sell when price below cloud + TK cross down). Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_filter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan = np.full(n, np.nan)
    for i in range(9, n):
        tenkan[i] = (np.max(high[i-9:i]) + np.min(low[i-9:i])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun = np.full(n, np.nan)
    for i in range(26, n):
        kijun[i] = (np.max(high[i-26:i]) + np.min(low[i-26:i])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = np.full(n, np.nan)
    for i in range(26, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_b = np.full(n, np.nan)
    for i in range(52, n):
        senkou_b[i] = (np.max(high[i-52:i]) + np.min(low[i-52:i])) / 2
    
    # Get 1d data for trend filter (price vs cloud)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku on 1d (same parameters)
    tenkan_1d = np.full(len(close_1d), np.nan)
    for i in range(9, len(close_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-9:i]) + np.min(low_1d[i-9:i])) / 2
    
    kijun_1d = np.full(len(close_1d), np.nan)
    for i in range(26, len(close_1d)):
        kijun_1d[i] = (np.max(high_1d[i-26:i]) + np.min(low_1d[i-26:i])) / 2
    
    senkou_a_1d = np.full(len(close_1d), np.nan)
    for i in range(26, len(close_1d)):
        if not np.isnan(tenkan_1d[i]) and not np.isnan(kijun_1d[i]):
            senkou_a_1d[i] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    senkou_b_1d = np.full(len(close_1d), np.nan)
    for i in range(52, len(close_1d)):
        senkou_b_1d[i] = (np.max(high_1d[i-52:i]) + np.min(low_1d[i-52:i])) / 2
    
    # 1d cloud top and bottom
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d Ichimoku to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    cloud_top_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    
    # 6h TK cross signals
    tk_cross_up = np.zeros(n, dtype=bool)
    tk_cross_down = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(tenkan[i-1]) and not np.isnan(kijun[i-1]) and not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            if tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]:
                tk_cross_up[i] = True
            if tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i]:
                tk_cross_down[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 60  # Need enough data for Ichimoku and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top_1d_aligned[i]) or np.isnan(cloud_bottom_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below 1d cloud bottom OR TK cross down
            # Stoploss: price drops 2*ATR below entry (using 6h ATR)
            # Calculate ATR for stoploss
            if i >= 1:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                # Simplified ATR approximation for stoploss
                atr_approx = tr  # Using current TR as proxy
                if (close[i] < cloud_bottom_1d_aligned[i] or
                    tk_cross_down[i] or
                    close[i] < entry_price - 2.0 * atr_approx):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price above 1d cloud top OR TK cross up
            # Stoploss: price rises 2*ATR above entry
            if i >= 1:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                atr_approx = tr
                if (close[i] > cloud_top_1d_aligned[i] or
                    tk_cross_up[i] or
                    close[i] > entry_price + 2.0 * atr_approx):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Entry conditions
                price_above_1d_cloud = close[i] > cloud_top_1d_aligned[i]
                price_below_1d_cloud = close[i] < cloud_bottom_1d_aligned[i]
                
                # Long: price above 1d cloud + TK cross up
                if price_above_1d_cloud and tk_cross_up[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: price below 1d cloud + TK cross down
                elif price_below_1d_cloud and tk_cross_down[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals