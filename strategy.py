#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrendFilter
Hypothesis: Trade 6h Ichimoku cloud breaks in direction of 1d trend (EMA50). 
Ichimoku provides objective support/resistance via cloud (Senkou Span A/B). 
Trend filter (1d EMA50) ensures we only take bullish breaks in uptrend, bearish breaks in downtrend. 
Avoids whipsaws in sideways markets by requiring cloud break + trend alignment.
Target: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag on 6h timeframe.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # The cloud is between Senkou Span A and Senkou Span B
    # Bullish when price is above cloud, bearish when below
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and 1d EMA50 (50)
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above cloud AND 1d trend bullish (close > EMA50)
            long_setup = (close[i] > cloud_top[i]) and \
                         (close_1d_align := align_htf_to_ltf(prices, df_1d, close_1d)[i]) > ema_50_1d_aligned[i]
            # Short: price breaks below cloud AND 1d trend bearish (close < EMA50)
            short_setup = (close[i] < cloud_bottom[i]) and \
                          (close_1d_align := align_htf_to_ltf(prices, df_1d, close_1d)[i]) < ema_50_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters cloud OR 1d trend turns bearish
            if (close[i] < cloud_top[i] and close[i] > cloud_bottom[i]) or \
               (align_htf_to_ltf(prices, df_1d, close_1d)[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters cloud OR 1d trend turns bullish
            if (close[i] < cloud_top[i] and close[i] > cloud_bottom[i]) or \
               (align_htf_to_ltf(prices, df_1d, close_1d)[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrendFilter"
timeframe = "6h"
leverage = 1.0