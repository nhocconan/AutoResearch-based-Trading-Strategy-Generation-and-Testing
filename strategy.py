#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend
Hypothesis: Use 6h timeframe with Ichimoku cloud and TK cross, filtered by 1d EMA50 trend.
Long when: price above cloud + TK cross bullish + 1d EMA50 uptrend.
Short when: price below cloud + TK cross bearish + 1d EMA50 downtrend.
Exit when: price crosses below/above cloud or TK cross reverses.
Ichimoku provides dynamic support/resistance and trend direction.
Works in both bull and bear via trend filter and cloud as volatility-adjusted bands.
Target: 12-37 trades/year for 6h timeframe.
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for entry)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Senkou B, 26 for cloud shift
    start_idx = 52 + 26  # 78 periods to have valid cloud data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        # Cloud boundaries (Senkou Span A and B plotted 26 periods ahead)
        # For current price, we use Senkou values from 26 periods ago
        idx_cloud = i - 26
        if idx_cloud < 0:
            signals[i] = 0.0
            continue
            
        span_a = senkou_span_a[idx_cloud]
        span_b = senkou_span_b[idx_cloud]
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # TK cross: Tenkan-sen crossing Kijun-sen
        tk_bullish = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_bearish = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        if position == 0:
            # Flat - look for entry with trend confirmation
            # Long: price above cloud + TK bullish cross + 1d EMA50 uptrend
            long_entry = (close_val > cloud_top) and tk_bullish and (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1])
            # Short: price below cloud + TK bearish cross + 1d EMA50 downtrend
            short_entry = (close_val < cloud_bottom) and tk_bearish and (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1])
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price crosses below cloud or TK turns bearish
            if (close_val < cloud_bottom) or (tenkan_sen[i] < kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price crosses above cloud or TK turns bullish
            if (close_val > cloud_top) or (tenkan_sen[i] > kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0