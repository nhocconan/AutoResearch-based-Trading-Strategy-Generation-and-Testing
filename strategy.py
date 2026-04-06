#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud strategy with daily timeframe filter
# Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) for trend direction
# Filters trades using daily timeframe cloud position to avoid counter-trend entries
# Designed for 6h timeframe to target 50-150 trades over 4 years with medium frequency
# Ichimoku works well in both trending and ranging markets, providing clear entry/exit signals

name = "6h_ichimoku1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1-day Ichimoku components for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    highest_high_tenkan = np.full(len(close_1d), np.nan)
    lowest_low_tenkan = np.full(len(close_1d), np.nan)
    tenkan_sen = np.full(len(close_1d), np.nan)
    
    for i in range(period_tenkan - 1, len(close_1d)):
        highest_high_tenkan[i] = np.max(high_1d[i - period_tenkan + 1:i + 1])
        lowest_low_tenkan[i] = np.min(low_1d[i - period_tenkan + 1:i + 1])
        tenkan_sen[i] = (highest_high_tenkan[i] + lowest_low_tenkan[i]) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    highest_high_kijun = np.full(len(close_1d), np.nan)
    lowest_low_kijun = np.full(len(close_1d), np.nan)
    kijun_sen = np.full(len(close_1d), np.nan)
    
    for i in range(period_kijun - 1, len(close_1d)):
        highest_high_kijun[i] = np.max(high_1d[i - period_kijun + 1:i + 1])
        lowest_low_kijun[i] = np.min(low_1d[i - period_kijun + 1:i + 1])
        kijun_sen[i] = (highest_high_kijun[i] + lowest_low_kijun[i]) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    highest_high_senkou_b = np.full(len(close_1d), np.nan)
    lowest_low_senkou_b = np.full(len(close_1d), np.nan)
    senkou_span_b = np.full(len(close_1d), np.nan)
    
    for i in range(period_senkou_b - 1, len(close_1d)):
        highest_high_senkou_b[i] = np.max(high_1d[i - period_senkou_b + 1:i + 1])
        lowest_low_senkou_b[i] = np.min(low_1d[i - period_senkou_b + 1:i + 1])
        senkou_span_b[i] = (highest_high_senkou_b[i] + lowest_low_senkou_b[i]) / 2
    
    # Shift Senkou Spans forward by 26 periods
    senkou_span_a_shifted = np.full(len(close_1d), np.nan)
    senkou_span_b_shifted = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d) - 26):
        senkou_span_a_shifted[i + 26] = senkou_span_a[i]
        senkou_span_b_shifted[i + 26] = senkou_span_b[i]
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_shifted)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted)
    
    # 6-hour Ichimoku components for entry signals
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    highest_high_tenkan_6h = np.full(n, np.nan)
    lowest_low_tenkan_6h = np.full(n, np.nan)
    tenkan_sen_6h = np.full(n, np.nan)
    
    for i in range(period_tenkan - 1, n):
        highest_high_tenkan_6h[i] = np.max(high[i - period_tenkan + 1:i + 1])
        lowest_low_tenkan_6h[i] = np.min(low[i - period_tenkan + 1:i + 1])
        tenkan_sen_6h[i] = (highest_high_tenkan_6h[i] + lowest_low_tenkan_6h[i]) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    highest_high_kijun_6h = np.full(n, np.nan)
    lowest_low_kijun_6h = np.full(n, np.nan)
    kijun_sen_6h = np.full(n, np.nan)
    
    for i in range(period_kijun - 1, n):
        highest_high_kijun_6h[i] = np.max(high[i - period_kijun + 1:i + 1])
        lowest_low_kijun_6h[i] = np.min(low[i - period_kijun + 1:i + 1])
        kijun_sen_6h[i] = (highest_high_kijun_6h[i] + lowest_low_kijun_6h[i]) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(period_kijun, period_senkou_b) + 26  # Ensure all Ichimoku components are available
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and price position relative to cloud
        # Green cloud: Senkou Span A > Senkou Span B (bullish)
        # Red cloud: Senkou Span A < Senkou Span B (bearish)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan/Kijun cross
        tk_cross_bullish = tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
        tk_cross_bearish = tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
        
        # Daily timeframe filter: only trade in direction of daily cloud
        daily_bullish = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        daily_bearish = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below Kijun-sen or enters opposite cloud
            if (close[i] < kijun_sen_6h[i] or 
                (daily_bearish and price_above_cloud)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Kijun-sen or enters opposite cloud
            if (close[i] > kijun_sen_6h[i] or 
                (daily_bullish and price_below_cloud)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries aligned with daily timeframe
            if daily_bullish:
                # Long: price above cloud + bullish TK cross
                if price_above_cloud and tk_cross_bullish:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            elif daily_bearish:
                # Short: price below cloud + bearish TK cross
                if price_below_cloud and tk_cross_bearish:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals