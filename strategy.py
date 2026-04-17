#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Daily Filter
Long: Price above 1d cloud + TK cross bullish on 6h
Short: Price below 1d cloud + TK cross bearish on 6h
Exit: TK cross reverses or price re-enters cloud
Uses Ichimoku components from daily timeframe for trend filter and 6h for entry timing
Target: 15-30 trades/year per symbol (60-120 total over 4 years)
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
    
    # Get 1d data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                     pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                    pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                        pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # The actual cloud is Senkou Span A and B shifted forward by 26 periods
    # For cloud at time t, we need values that were calculated 26 periods ago
    senkou_span_a_1d_shifted = np.roll(senkou_span_a_1d, -kijun_period)
    senkou_span_b_1d_shifted = np.roll(senkou_span_b_1d, -kijun_period)
    
    # Handle the end where we don't have future values
    senkou_span_a_1d_shifted[-kijun_period:] = np.nan
    senkou_span_b_1d_shifted[-kijun_period:] = np.nan
    
    # Align Ichimoku components to 6h
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d_shifted)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d_shifted)
    
    # Calculate 6h TK cross for entry timing
    # Tenkan-sen and Kijun-sen on 6h
    tenkan_sen_6h = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                     pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun_sen_6h = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                    pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    tk_cross_6h = tenkan_sen_6h - kijun_sen_6h  # Positive when bullish
    tk_cross_prev_6h = np.roll(tk_cross_6h, 1)
    tk_cross_prev_6h[0] = 0
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(100, kijun_period + senkou_span_b_period)  # need sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tk_cross_6h[i]) or np.isnan(tk_cross_prev_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan_1d = tenkan_sen_1d_aligned[i]
        kijun_1d = kijun_sen_1d_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        tk_now = tk_cross_6h[i]
        tk_prev = tk_cross_prev_6h[i]
        
        # Cloud boundaries (use the higher as top, lower as bottom)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Price above/below cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        if position == 0:
            # Long: Price above cloud + TK cross turns bullish
            if price_above_cloud and tk_prev <= 0 and tk_now > 0:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + TK cross turns bearish
            elif price_below_cloud and tk_prev >= 0 and tk_now < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross turns bearish OR price re-enters cloud
            if tk_now < 0 or (price <= cloud_top and price >= cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross turns bullish OR price re-enters cloud
            if tk_now > 0 or (price <= cloud_top and price >= cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_DailyFilter_TKCross"
timeframe = "6h"
leverage = 1.0