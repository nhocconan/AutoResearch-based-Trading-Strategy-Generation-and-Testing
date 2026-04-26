#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. In 6h timeframe, price breaking above/below cloud with weekly trend filter (price > weekly EMA50 for longs, < weekly EMA50 for shorts) captures strong momentum moves. Weekly EMA50 filter ensures alignment with higher timeframe trend, reducing whipsaws in ranging markets. Uses discrete position sizing (0.25) to control drawdown and minimize fee churn. Designed to work in both bull (breakouts with trend) and bear (trend continuation after pullbacks to cloud) markets.
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
    
    # Pre-compute session filter (UTC 8-20) for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get weekly data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations (52 for Senkou Span B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or
            not in_session[i]):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_1w_val = ema_50_1w_aligned[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        span_a_val = senkou_span_a_aligned[i]
        span_b_val = senkou_span_b_aligned[i]
        close_val = close[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(span_a_val, span_b_val)
        cloud_bottom = min(span_a_val, span_b_val)
        
        if position == 0:
            # Long: price breaks above cloud AND weekly trend is up (close > weekly EMA50)
            long_signal = (close_val > cloud_top) and (close_val > ema_50_1w_val)
            # Short: price breaks below cloud AND weekly trend is down (close < weekly EMA50)
            short_signal = (close_val < cloud_bottom) and (close_val < ema_50_1w_val)
            
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
            # Exit: price falls below cloud bottom (cloud support broken)
            if close_val < cloud_bottom:
                signals[i] = 0.0
                position = 0
            # Exit: weekly trend reverses (close < weekly EMA50)
            elif close_val < ema_50_1w_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price rises above cloud top (cloud resistance broken)
            if close_val > cloud_top:
                signals[i] = 0.0
                position = 0
            # Exit: weekly trend reverses (close > weekly EMA50)
            elif close_val > ema_50_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0