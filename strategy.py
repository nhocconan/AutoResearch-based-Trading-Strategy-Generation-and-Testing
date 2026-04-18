#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_1DTrend_Filter
Hypothesis: Ichimoku cloud on 6h provides clear support/resistance and trend direction, with daily timeframe filtering to avoid false signals. 
Long when Tenkan crosses above Kijun and price is above cloud, with daily trend confirmation.
Short when Tenkan crosses below Kijun and price is below cloud, with daily trend confirmation.
This strategy works in both bull and bear markets by following the dominant trend on higher timeframe.
Target: 15-35 trades/year by requiring multiple confirmations (TK cross, cloud position, daily trend).
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
    
    # Ichimoku parameters (standard)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_high_9 = np.full_like(high, np.nan)
    lowest_low_9 = np.full_like(low, np.nan)
    for i in range(tenkan_period - 1, len(high)):
        highest_high_9[i] = np.max(high[i - tenkan_period + 1:i + 1])
        lowest_low_9[i] = np.min(low[i - tenkan_period + 1:i + 1])
    tenkan = (highest_high_9 + lowest_low_9) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_high_26 = np.full_like(high, np.nan)
    lowest_low_26 = np.full_like(low, np.nan)
    for i in range(kijun_period - 1, len(high)):
        highest_high_26[i] = np.max(high[i - kijun_period + 1:i + 1])
        lowest_low_26[i] = np.min(low[i - kijun_period + 1:i + 1])
    kijun = (highest_high_26 + lowest_low_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = (tenkan + kijun) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
    highest_high_52 = np.full_like(high, np.nan)
    lowest_low_52 = np.full_like(low, np.nan)
    for i in range(senkou_span_b_period - 1, len(high)):
        highest_high_52[i] = np.max(high[i - senkou_span_b_period + 1:i + 1])
        lowest_low_52[i] = np.min(low[i - senkou_span_b_period + 1:i + 1])
    senkou_span_b = (highest_high_52 + lowest_low_52) / 2
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily for trend filter
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # Align Ichimoku components and daily EMA to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), tenkan)
    kijun_6h = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kijun)
    senkou_span_a_6h = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_span_b)
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + 26  # Ensure we have enough data for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or 
            np.isnan(ema_50_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        lower_cloud = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # TK Cross signals
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        # Price position relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Daily trend filter
        daily_uptrend = close[i] > ema_50_1d_6h[i]
        daily_downtrend = close[i] < ema_50_1d_6h[i]
        
        if position == 0:
            # Long: TK cross up, price above cloud, daily uptrend
            if tk_cross_up and price_above_cloud and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down, price below cloud, daily downtrend
            elif tk_cross_down and price_below_cloud and daily_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross down OR price falls below cloud
            if tk_cross_down or close[i] < upper_cloud:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross up OR price rises above cloud
            if tk_cross_up or close[i] > lower_cloud:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1DTrend_Filter"
timeframe = "6h"
leverage = 1.0