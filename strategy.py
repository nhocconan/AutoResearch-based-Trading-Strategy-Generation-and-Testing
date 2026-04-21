#!/usr/bin/env python3
"""
6h_1d_IchiCloud_TK_Cross_TrendFilter
Hypothesis: Ichimoku cloud with TK cross on 6h timeframe, filtered by 1d cloud color (bull/bear regime), works in both bull and bear markets by only taking trades aligned with the higher timeframe trend. Uses discrete position sizing (0.25) to limit fee drag and target 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for HTF Ichimoku cloud (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align 1d Ichimoku components to 6h timeframe (cloud for regime filter)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate Ichimoku on 6h timeframe for entry signals (TK cross)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (9-period) on 6h
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (26-period) on 6h
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # 1d Cloud color (regime filter): bullish if Senkou Span A > Senkou Span B
        bullish_regime = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        bearish_regime = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        # 6h TK Cross
        tk_cross_bull = tenkan_sen_6h[i] > kijun_sen_6h[i]
        tk_cross_bear = tenkan_sen_6h[i] < kijun_sen_6h[i]
        
        if position == 0:
            # Long: bullish regime + bullish TK cross
            if bullish_regime and tk_cross_bull:
                signals[i] = 0.25
                position = 1
            # Short: bearish regime + bearish TK cross
            elif bearish_regime and tk_cross_bear:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish TK cross or price breaks below cloud
            if tk_cross_bear or price < senkou_span_a_aligned[i] or price < senkou_span_b_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish TK cross or price breaks above cloud
            if tk_cross_bull or price > senkou_span_a_aligned[i] or price > senkou_span_b_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_IchiCloud_TK_Cross_TrendFilter"
timeframe = "6h"
leverage = 1.0