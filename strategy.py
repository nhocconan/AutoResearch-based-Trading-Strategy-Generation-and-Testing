#!/usr/bin/env python3
"""
1d_Ichimoku_TK_Cross_CloudFilter_WeeklyTrend
Hypothesis: Trade Ichimoku TK cross (Tenkan/Kijun) on daily timeframe with cloud filter and weekly trend confirmation.
In bull markets (price > weekly cloud): go long on TK cross up, short on TK cross down.
In bear markets (price < weekly cloud): only short on TK cross down, exit on TK cross up.
Weekly trend filter reduces whipsaw in sideways markets. Uses volume > 1.5x 20-day average for confirmation.
Designed for low trade frequency (<20/year) to minimize fee drag while capturing major trends.
Works in both bull and bear by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = np.full_like(close_1d, np.nan)
    if len(high_1d) >= tenkan_period:
        for i in range(tenkan_period - 1, len(high_1d)):
            tenkan[i] = (np.max(high_1d[i - tenkan_period + 1:i + 1]) + 
                         np.min(low_1d[i - tenkan_period + 1:i + 1])) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = np.full_like(close_1d, np.nan)
    if len(high_1d) >= kijun_period:
        for i in range(kijun_period - 1, len(high_1d)):
            kijun[i] = (np.max(high_1d[i - kijun_period + 1:i + 1]) + 
                        np.min(low_1d[i - kijun_period + 1:i + 1])) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = np.full_like(close_1d, np.nan)
    if len(tenkan) >= kijun_period and len(kijun) >= kijun_period:
        for i in range(len(tenkan)):
            if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                idx = i + kijun_period
                if idx < len(senkou_span_a):
                    senkou_span_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = np.full_like(close_1d, np.nan)
    if len(high_1d) >= senkou_span_b_period:
        for i in range(senkou_span_b_period - 1, len(high_1d)):
            span_b_val = (np.max(high_1d[i - senkou_span_b_period + 1:i + 1]) + 
                          np.min(low_1d[i - senkou_span_b_period + 1:i + 1])) / 2
            idx = i + kijun_period
            if idx < len(senkou_span_b):
                senkou_span_b[idx] = span_b_val
    
    # Align Ichimoku components to daily timeframe (same as input)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Ichimoku cloud for trend filter
    tenkan_w = np.full_like(close_1w, np.nan)
    kijun_w = np.full_like(close_1w, np.nan)
    senkou_span_a_w = np.full_like(close_1w, np.nan)
    senkou_span_b_w = np.full_like(close_1w, np.nan)
    
    if len(high_1w) >= tenkan_period:
        for i in range(tenkan_period - 1, len(high_1w)):
            tenkan_w[i] = (np.max(high_1w[i - tenkan_period + 1:i + 1]) + 
                           np.min(low_1w[i - tenkan_period + 1:i + 1])) / 2
    
    if len(high_1w) >= kijun_period:
        for i in range(kijun_period - 1, len(high_1w)):
            kijun_w[i] = (np.max(high_1w[i - kijun_period + 1:i + 1]) + 
                          np.min(low_1w[i - kijun_period + 1:i + 1])) / 2
    
    if len(tenkan_w) >= kijun_period and len(kijun_w) >= kijun_period:
        for i in range(len(tenkan_w)):
            if not np.isnan(tenkan_w[i]) and not np.isnan(kijun_w[i]):
                idx = i + kijun_period
                if idx < len(senkou_span_a_w):
                    senkou_span_a_w[idx] = (tenkan_w[i] + kijun_w[i]) / 2
    
    if len(high_1w) >= senkou_span_b_period:
        for i in range(senkou_span_b_period - 1, len(high_1w)):
            span_b_val = (np.max(high_1w[i - senkou_span_b_period + 1:i + 1]) + 
                          np.min(low_1w[i - senkou_span_b_period + 1:i + 1])) / 2
            idx = i + kijun_period
            if idx < len(senkou_span_b_w):
                senkou_span_b_w[idx] = span_b_val
    
    # Align weekly Ichimoku to daily timeframe
    tenkan_w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_w)
    kijun_w_aligned = align_htf_to_ltf(prices, df_1w, kijun_w)
    senkou_span_a_w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_w)
    senkou_span_b_w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(
        tenkan_period + kijun_period,  # Ichomoku warmup
        kijun_period + senkou_span_b_period,  # for cloud
        vol_period
    )
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_w_aligned[i]) or np.isnan(kijun_w_aligned[i]) or
            np.isnan(senkou_span_a_w_aligned[i]) or np.isnan(senkou_span_b_w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Determine weekly cloud boundaries for trend filter
        weekly_cloud_top = max(senkou_span_a_w_aligned[i], senkou_span_b_w_aligned[i])
        weekly_cloud_bottom = min(senkou_span_a_w_aligned[i], senkou_span_b_w_aligned[i])
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # TK cross signals
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        if position == 0:
            # Determine market regime using weekly trend
            # Bull: price above weekly cloud
            # Bear: price below weekly cloud
            is_bull = close[i] > weekly_cloud_top
            is_bear = close[i] < weekly_cloud_bottom
            
            if is_bull and tk_cross_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            elif is_bear and tk_cross_down and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross down or price touches weekly cloud bottom (stop)
            if tk_cross_down or close[i] <= weekly_cloud_bottom:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross up or price touches weekly cloud top (stop)
            if tk_cross_up or close[i] >= weekly_cloud_top:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Ichimoku_TK_Cross_CloudFilter_WeeklyTrend"
timeframe = "1d"
leverage = 1.0