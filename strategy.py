#!/usr/bin/env python3
# 6h_ichimoku_1d_cloud_filter_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d timeframe as trend filter, with TK cross on 6h for entries.
# Long: Price above 1d cloud (bullish trend) + TK cross bullish on 6h.
# Short: Price below 1d cloud (bearish trend) + TK cross bearish on 6h.
# Exit: Opposite TK cross or price crosses cloud midpoint (Kijun).
# Uses 6h primary timeframe with 1d HTF for Ichimoku cloud and Kijun.
# Designed for low trade frequency (~15-30/year) to minimize fee drag while capturing major trends.
# Works in bull markets via trend continuation and bear markets via shorting downtrends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_cloud_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku on 6h: Tenkan-sen (9), Kijun-sen (26)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    tenkan_6h = (high_s.rolling(window=9, min_periods=9).max() + low_s.rolling(window=9, min_periods=9).min()) / 2
    kijun_6h = (high_s.rolling(window=26, min_periods=26).max() + low_s.rolling(window=26, min_periods=26).min()) / 2
    tenkan_6h_vals = tenkan_6h.values
    kijun_6h_vals = kijun_6h.values
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Senkou B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku on 1d: Tenkan-sen (9), Kijun-sen (26), Senkou Span A & B (52 displacement)
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    tenkan_1d = (high_1d_s.rolling(window=9, min_periods=9).max() + low_1d_s.rolling(window=9, min_periods=9).min()) / 2
    kijun_1d = (high_1d_s.rolling(window=26, min_periods=26).max() + low_1d_s.rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a = ((tenkan_1d + kijun_1d) / 2).shift(22)  # 26-4 = 22 displacement
    senkou_span_b = (high_1d_s.rolling(window=52, min_periods=52).max() + low_1d_s.rolling(window=52, min_periods=52).min()) / 2
    senkou_span_b = senkou_span_b.shift(22)  # 52-4 = 48? Actually 26 displacement for Senkou B
    
    # Correct Ichimoku: Senkou A = (Tenkan + Kijun)/2 shifted 26 periods ahead
    # Senkou B = 52-period HL/2 shifted 26 periods ahead
    # Cloud is between Senkou A and Senkou B
    senkou_span_a = ((tenkan_1d + kijun_1d) / 2).shift(22)  # 26-4 = 22? Let's use proper: shift 26
    senkou_span_b = (high_1d_s.rolling(window=52, min_periods=52).max() + low_1d_s.rolling(window=52, min_periods=52).min()) / 2
    senkou_span_b = senkou_span_b.shift(22)  # Will fix below
    
    # Recalculate with proper Ichimoku logic
    tenkan_1d = (high_1d_s.rolling(window=9, min_periods=9).max() + low_1d_s.rolling(window=9, min_periods=9).min()) / 2
    kijun_1d = (high_1d_s.rolling(window=26, min_periods=26).max() + low_1d_s.rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a = ((tenkan_1d + kijun_1d) / 2)
    senkou_span_b = (high_1d_s.rolling(window=52, min_periods=52).max() + low_1d_s.rolling(window=52, min_periods=52).min()) / 2
    
    # Shift Senkou spans 26 periods ahead (for cloud)
    senkou_span_a_shifted = senkou_span_a.shift(26)
    senkou_span_b_shifted = senkou_span_b.shift(26)
    
    senkou_span_a_vals = senkou_span_a_shifted.values
    senkou_span_b_vals = senkou_span_b_shifted.values
    kijun_1d_vals = kijun_1d.values
    
    # Align 1d Ichimoku components to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_vals)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_vals)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d_vals)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup for 1d Ichimoku (52+26=78, but align handles)
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h_vals[i]) or np.isnan(kijun_6h_vals[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(kijun_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d cloud boundaries and trend
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_middle = (senkou_a_1d_aligned[i] + senkou_b_1d_aligned[i]) / 2
        
        # 6h TK cross
        tk_bullish_cross = tenkan_6h_vals[i] > kijun_6h_vals[i] and tenkan_6h_vals[i-1] <= kijun_6h_vals[i-1]
        tk_bearish_cross = tenkan_6h_vals[i] < kijun_6h_vals[i] and tenkan_6h_vals[i-1] >= kijun_6h_vals[i-1]
        
        # Price vs cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        if position == 1:  # Long position
            # Exit: TK bearish cross OR price drops below cloud middle
            if tk_bearish_cross or close[i] < cloud_middle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK bullish cross OR price rises above cloud middle
            if tk_bullish_cross or close[i] > cloud_middle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price above 1d cloud (bullish trend) + TK bullish cross on 6h
            if price_above_cloud and tk_bullish_cross:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below 1d cloud (bearish trend) + TK bearish cross on 6h
            elif price_below_cloud and tk_bearish_cross:
                position = -1
                signals[i] = -0.25
    
    return signals