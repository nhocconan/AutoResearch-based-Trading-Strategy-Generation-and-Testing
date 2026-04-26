#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_12hTrend_v1
Hypothesis: Ichimoku cloud breakout (price above/below cloud) with 12h trend filter (price vs 12h EMA50) captures strong momentum moves while avoiding sideways whipsaws. Works in bull/bear via 12h trend alignment. Designed for 6h to target 12-37 trades/year with discrete sizing (0.25).
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Current cloud boundaries (Senkou Span A/B from 26 periods ago)
    senkou_span_a_lagged = np.roll(senkou_span_a, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b, 26)
    senkou_span_a_lagged[:26] = np.nan
    senkou_span_b_lagged[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_span_a_lagged, senkou_span_b_lagged)
    cloud_bottom = np.minimum(senkou_span_a_lagged, senkou_span_b_lagged)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Ichimoku calculations (52 + 26 for cloud shift)
    start_idx = 52 + 26
    
    for i in range(start_idx, n):
        close_val = close[i]
        ema_val = ema_50_12h_aligned[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(cloud_top_val) or np.isnan(cloud_bottom_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs 12h EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price ABOVE cloud with 12h uptrend
        long_condition = (close_val > cloud_top_val) and uptrend
        # Short: price BELOW cloud with 12h downtrend
        short_condition = (close_val < cloud_bottom_val) and downtrend
        
        # Exit: price re-enters cloud
        long_exit = (position == 1 and close_val <= cloud_top_val)
        short_exit = (position == -1 and close_val >= cloud_bottom_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_12hTrend_v1"
timeframe = "6h"
leverage = 1.0