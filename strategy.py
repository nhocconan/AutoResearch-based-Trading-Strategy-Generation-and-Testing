#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_Filter
Hypothesis: Ichimoku cloud breakout on 6h with 1w trend filter (price above/below weekly Kumo) captures strong momentum moves in both bull/bear markets. Uses Kumo twist for trend confirmation and discrete sizing (0.25) to limit fee drag. Targets 15-30 trades/year by requiring confluence of price breaking cloud, TK cross alignment, and weekly trend filter.
"""

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
    
    # Get 1w data for HTF trend filter (weekly Kumo)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou = np.concatenate([np.full(26, np.nan), close[:-26]]) if len(close) >= 26 else np.full(len(close), np.nan)
    
    # Align Ichimoku components to avoid look-ahead (Senkou spans are already plotted ahead)
    # For Senkou spans, we need to shift back by 26 periods to get current values
    senkou_span_a_current = np.concatenate([np.full(26, np.nan), senkou_span_a[:-26]]) if len(senkou_span_a) >= 26 else np.full(len(senkou_span_a), np.nan)
    senkou_span_b_current = np.concatenate([np.full(26, np.nan), senkou_span_b[:-26]]) if len(senkou_span_b) >= 26 else np.full(len(senkou_span_b), np.nan)
    
    # Get weekly Kumo (cloud) from 1w data
    # Weekly Tenkan-sen
    max_high_1w_tenkan = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    min_low_1w_tenkan = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (max_high_1w_tenkan + min_low_1w_tenkan) / 2
    
    # Weekly Kijun-sen
    max_high_1w_kijun = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    min_low_1w_kijun = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (max_high_1w_kijun + min_low_1w_kijun) / 2
    
    # Weekly Senkou Span A
    senkou_span_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    # Weekly Senkou Span B
    max_high_1w_senkou_b = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    min_low_1w_senkou_b = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1w = ((max_high_1w_senkou_b + min_low_1w_senkou_b) / 2)
    
    # Align weekly Kumo to 6h timeframe (using previous completed weekly bar)
    prev_senkou_span_a_1w = np.concatenate([[np.nan], senkou_span_a_1w[:-1]])
    prev_senkou_span_b_1w = np.concatenate([[np.nan], senkou_span_b_1w[:-1]])
    
    # Align all indicators
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_span_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a_current)
    senkou_span_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b_current)
    chikou_aligned = align_htf_to_ltf(prices, prices, chikou)
    
    # Align weekly Kumo (using previous bar to avoid look-ahead)
    senkou_span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, prev_senkou_span_a_1w)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, prev_senkou_span_b_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(52, 26, 9)  # Senkou B, Kijun, Tenkan
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(senkou_span_a_1w_aligned[i]) or 
            np.isnan(senkou_span_b_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        span_a_val = senkou_span_a_aligned[i]
        span_b_val = senkou_span_b_aligned[i]
        span_a_1w_val = senkou_span_a_1w_aligned[i]
        span_b_1w_val = senkou_span_b_1w_aligned[i]
        close_val = close[i]
        
        # Determine cloud boundaries (top and bottom of Kumo)
        cloud_top = max(span_a_val, span_b_val)
        cloud_bottom = min(span_a_val, span_b_val)
        
        # Determine weekly cloud boundaries
        weekly_cloud_top = max(span_a_1w_val, span_b_1w_val)
        weekly_cloud_bottom = min(span_a_1w_val, span_b_1w_val)
        
        # Weekly trend filter: price above/below weekly cloud
        weekly_uptrend = close_val > weekly_cloud_top
        weekly_downtrend = close_val < weekly_cloud_bottom
        
        # TK cross conditions
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        # Price position relative to cloud
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        if position == 0:
            # Look for entry signals: Kumo breakout with TK cross alignment and weekly trend
            # Long: price breaks above cloud with bullish TK cross and weekly uptrend
            long_signal = price_above_cloud and tk_bullish and weekly_uptrend
            # Short: price breaks below cloud with bearish TK cross and weekly downtrend
            short_signal = price_below_cloud and tk_bearish and weekly_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price re-enters cloud (break of Kumo)
            # 2. TK cross turns bearish
            if price_below_cloud or not tk_bullish:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price re-enters cloud (break of Kumo)
            # 2. TK cross turns bullish
            if price_above_cloud or not tk_bearish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0