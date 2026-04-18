#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend
6h strategy using Ichimoku cloud and TK cross with weekly trend filter.
- Long: Tenkan-sen crosses above Kijun-sen AND price above cloud AND weekly trend up
- Short: Tenkan-sen crosses below Kijun-sen AND price below cloud AND weekly trend down
- Exit: Opposite TK cross or price crosses opposite cloud boundary
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (trend following) and bear markets (counter-trend reversals)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (
        pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values +
        pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    ) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (
        pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values +
        pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    ) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (
        pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values +
        pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    ) / 2
    
    # Chikou Span (Lagging Span): not used in signals
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA25 for trend filter
    ema_25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_aligned = align_htf_to_ltf(prices, df_1w, ema_25_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough for Ichimoku calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_25_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku conditions
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Cloud boundaries
        upper_cloud = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_25_aligned[i]
        weekly_downtrend = close[i] < ema_25_aligned[i]
        
        if position == 0:
            # Long: TK cross up + price above cloud + weekly uptrend
            if tk_cross_up and price_above_cloud and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + weekly downtrend
            elif tk_cross_down and price_below_cloud and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross down OR price crosses below cloud
            if tk_cross_down or close[i] < lower_cloud:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross up OR price crosses above cloud
            if tk_cross_up or close[i] > upper_cloud:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend"
timeframe = "6h"
leverage = 1.0