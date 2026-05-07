#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_1dTrend_Trend_Continuation_v1
# Hypothesis: Uses Ichimoku cloud (from 1d) as primary trend filter and TK cross from 1d for entry signals.
# In bull markets: price above 1d cloud + TK cross bullish = long.
# In bear markets: price below 1d cloud + TK cross bearish = short.
# The 1d Ichimoku provides robust trend identification and dynamic support/resistance,
# reducing whipsaws in sideways markets while capturing trends. TK cross provides timely
# entry signals aligned with the higher timeframe trend.
# Target: 20-40 trades/year to minimize fee drag while maintaining edge.

name = "6h_Ichimoku_Cloud_1dTrend_Trend_Continuation_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku (26*2)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
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
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        lower_cloud = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # TK cross signals
        tk_cross_bullish = tenkan_sen_6h[i] > kijun_sen_6h[i]
        tk_cross_bearish = tenkan_sen_6h[i] < kijun_sen_6h[i]
        
        if position == 0:
            # Long: Price above cloud + TK cross bullish
            if close[i] > upper_cloud and tk_cross_bullish:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + TK cross bearish
            elif close[i] < lower_cloud and tk_cross_bearish:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below cloud or TK cross bearish
            if close[i] < lower_cloud or tk_cross_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above cloud or TK cross bullish
            if close[i] > upper_cloud or tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals