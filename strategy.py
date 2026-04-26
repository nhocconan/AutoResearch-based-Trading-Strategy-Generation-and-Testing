#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter
Hypothesis: Ichimoku cloud breakouts on 6h with weekly trend filter (price above/below weekly Kumo) and volume confirmation (>1.5x 24-period average). Uses discrete sizing 0.25 to target ~25 trades/year (100 total over 4 years). Designed for both bull and bear markets: weekly trend filter ensures alignment with higher timeframe momentum, Ichimoku provides objective support/resistance levels, and volume confirmation filters weak breakouts. The cloud acts as dynamic support/resistance, reducing whipsaws in ranging markets.
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
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: UTC 0-23 (trade all sessions for 6h to capture global moves)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = np.ones(n, dtype=bool)  # 6h: trade all sessions
    
    # Get weekly data for trend filter (Kumo cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 1 year of weekly data
        return np.zeros(n)
    
    # Weekly Ichimoku components for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    if len(high_1w) < 52:
        senkou_span_a_1w = np.full_like(close_1w, np.nan)
        senkou_span_b_1w = np.full_like(close_1w, np.nan)
        chikou_span_1w = np.full_like(close_1w, np.nan)
    else:
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
        senkou_span_b = (period52_high + period52_low) / 2
        
        # Chikou Span (Lagging Span): Close shifted 26 periods behind
        chikou_span = close_1w
        
        # Shift Senkou Spans ahead by 26 periods for cloud calculation
        senkou_span_a_1w = np.concatenate([np.full(26, np.nan), senkou_span_a[:-26]]) if len(senkou_span_a) > 26 else np.full_like(close_1w, np.nan)
        senkou_span_b_1w = np.concatenate([np.full(26, np.nan), senkou_span_b[:-26]]) if len(senkou_span_b) > 26 else np.full_like(close_1w, np.nan)
        chikou_span_1w = np.concatenate([np.full(26, np.nan), chikou_span[:-26]]) if len(chikou_span) > 26 else np.full_like(close_1w, np.nan)
    
    # Align weekly Ichimoku to 6h
    senkou_span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w)
    chikou_span_1w_aligned = align_htf_to_ltf(prices, df_1w, chikou_span_1w)
    
    # Get 6h data for Ichimoku entry signals
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    # 6h Ichimoku components
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    if len(high_6h) < 52:
        tenkan_sen_6h = np.full_like(close_6h, np.nan)
        kijun_sen_6h = np.full_like(close_6h, np.nan)
        senkou_span_a_6h = np.full_like(close_6h, np.nan)
        senkou_span_b_6h = np.full_like(close_6h, np.nan)
        chikou_span_6h = np.full_like(close_6h, np.nan)
    else:
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high_6h = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
        period9_low_6h = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
        tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high_6h = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
        period26_low_6h = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
        kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        senkou_span_a_6h = (tenkan_sen_6h + kijun_sen_6h) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high_6h = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
        period52_low_6h = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
        senkou_span_b_6h = (period52_high_6h + period52_low_6h) / 2
        
        # Chikou Span (Lagging Span): Close shifted 26 periods behind
        chikou_span_6h = close_6h
        
        # Shift Senkou Spans ahead by 26 periods for cloud calculation
        senkou_span_a_6h = np.concatenate([np.full(26, np.nan), senkou_span_a_6h[:-26]]) if len(senkou_span_a_6h) > 26 else np.full_like(close_6h, np.nan)
        senkou_span_b_6h = np.concatenate([np.full(26, np.nan), senkou_span_b_6h[:-26]]) if len(senkou_span_b_6h) > 26 else np.full_like(close_6h, np.nan)
        chikou_span_6h = np.concatenate([np.full(26, np.nan), chikou_span_6h[:-26]]) if len(chikou_span_6h) > 26 else np.full_like(close_6h, np.nan)
    
    # Align 6h Ichimoku to prices (6h is primary timeframe, so direct alignment)
    # For 6h timeframe, we need to align the 6h Ichimoku to the 6h prices
    # Since prices is already 6h, we can use the values directly with proper indexing
    # But we need to account for the 26-period shift in Senkou Span
    tenkan_sen_6h_aligned = tenkan_sen_6h
    kijun_sen_6h_aligned = kijun_sen_6h
    senkou_span_a_6h_aligned = senkou_span_a_6h
    senkou_span_b_6h_aligned = senkou_span_b_6h
    chikou_span_6h_aligned = chikou_span_6h
    
    # Volume average (24-period = 4 days of 6h data) for confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly Ichimoku (52), 6h Ichimoku (52), volume MA (24)
    start_idx = max(52, 52, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(senkou_span_a_1w_aligned[i]) or 
            np.isnan(senkou_span_b_1w_aligned[i]) or
            np.isnan(tenkan_sen_6h_aligned[i]) or
            np.isnan(kijun_sen_6h_aligned[i]) or
            np.isnan(senkou_span_a_6h_aligned[i]) or
            np.isnan(senkou_span_b_6h_aligned[i]) or
            np.isnan(chikou_span_6h_aligned[i]) or
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter: price relative to weekly Kumo (cloud)
        weekly_kumo_top = max(senkou_span_a_1w_aligned[i], senkou_span_b_1w_aligned[i])
        weekly_kumo_bottom = min(senkou_span_a_1w_aligned[i], senkou_span_b_1w_aligned[i])
        price_above_weekly_kumo = close[i] > weekly_kumo_top
        price_below_weekly_kumo = close[i] < weekly_kumo_bottom
        
        # 6h Ichimoku signals
        tenkan = tenkan_sen_6h_aligned[i]
        kijun = kijun_sen_6h_aligned[i]
        senkou_a = senkou_span_a_6h_aligned[i]
        senkou_b = senkou_span_b_6h_aligned[i]
        chikou = chikou_span_6h_aligned[i]
        
        # 6h Kumo (cloud)
        kumo_top = max(senkou_a, senkou_b)
        kumo_bottom = min(senkou_a, senkou_b)
        
        # Price relative to 6h Kumo
        price_above_6h_kumo = close[i] > kumo_top
        price_below_6h_kumo = close[i] < kumo_bottom
        
        # TK Cross (Tenkan-sen crossing Kijun-sen)
        tk_cross_up = tenkan > kijun and tenkan_sen_6h_aligned[i-1] <= kijun_sen_6h_aligned[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen_6h_aligned[i-1] >= kijun_sen_6h_aligned[i-1]
        
        # Volume confirmation: current volume > 1.5x 24-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above weekly Kumo AND price above 6h Kumo AND TK cross up AND volume confirmation
            long_signal = (price_above_weekly_kumo and 
                          price_above_6h_kumo and 
                          tk_cross_up and 
                          volume_confirmed)
            # Short: price below weekly Kumo AND price below 6h Kumo AND TK cross down AND volume confirmation
            short_signal = (price_below_weekly_kumo and 
                           price_below_6h_kumo and 
                           tk_cross_down and 
                           volume_confirmed)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below 6h Kumo OR TK cross down
            if close[i] < kumo_bottom or tk_cross_down:
                signals[i] = 0.0
                position = 0
            # Exit: weekly trend reversal (price below weekly Kumo)
            elif price_below_weekly_kumo:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above 6h Kumo OR TK cross up
            if close[i] > kumo_top or tk_cross_up:
                signals[i] = 0.0
                position = 0
            # Exit: weekly trend reversal (price above weekly Kumo)
            elif price_above_weekly_kumo:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0