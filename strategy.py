#!/usr/bin/env python3
"""
6h_Ichimoku_Kijun_Bounce_1dTrendFilter
Hypothesis: Trade price bounces off the Ichimoku Kijun-sen (26-period) on 6h with 1d trend filter. In bull markets (price > 1d Kumo), go long on Kijun bounce; in bear markets (price < 1d Kumo), short on Kijun bounce. Uses Kumo twist for early trend change detection. Target: 15-25 trades/year per symbol.
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
    
    # Get 1d data for HTF trend filter (Kumo)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Kumo calculation
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
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
    
    # Get 1d Ichimoku for trend filter (Kumo)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # 1d Tenkan-sen (9-period)
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    # 1d Kijun-sen (26-period)
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    # 1d Senkou Span A
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    
    # 1d Senkou Span B (52-period)
    period52_high_1d = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = ((period52_high_1d + period52_low_1d) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations
    start_idx = 52  # for 52-period calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d Kumo (cloud) trend
        # Bullish Kumo: Senkou Span A > Senkou Span B
        # Bearish Kumo: Senkou Span A < Senkou Span B
        bullish_kumo = senkou_span_a_1d_aligned[i] > senkou_span_b_1d_aligned[i]
        bearish_kumo = senkou_span_a_1d_aligned[i] < senkou_span_b_1d_aligned[i]
        
        # Kumo twist detection (early trend change signal)
        # Bullish twist: prior Senkou A < prior Senkou B, current A > current B
        # Bearish twist: prior Senkou A > prior Senkou B, current A < current B
        bullish_twist = False
        bearish_twist = False
        if i > start_idx:
            bullish_twist = (senkou_span_a_1d_aligned[i-1] < senkou_span_b_1d_aligned[i-1] and 
                           senkou_span_a_1d_aligned[i] > senkou_span_b_1d_aligned[i])
            bearish_twist = (senkou_span_a_1d_aligned[i-1] > senkou_span_b_1d_aligned[i-1] and 
                           senkou_span_a_1d_aligned[i] < senkou_span_b_1d_aligned[i])
        
        if position == 0:
            # Long setup: price near Kijun-sen (within 0.5%) + bullish Kumo OR Kumo twist
            kijun_distance = abs(close[i] - kijun_sen[i]) / kijun_sen[i]
            long_setup = (kijun_distance < 0.005) and (bullish_kumo or bullish_twist)
            
            # Short setup: price near Kijun-sen (within 0.5%) + bearish Kumo OR Kumo twist
            short_setup = (kijun_distance < 0.005) and (bearish_kumo or bearish_twist)
            
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
            # Exit: price moves away from Kijun-sen (>1.0%) OR bearish Kumo established
            kijun_distance = abs(close[i] - kijun_sen[i]) / kijun_sen[i]
            if (kijun_distance > 0.01) or bearish_kumo:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price moves away from Kijun-sen (>1.0%) OR bullish Kumo established
            kijun_distance = abs(close[i] - kijun_sen[i]) / kijun_sen[i]
            if (kijun_distance > 0.01) or bullish_kumo:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kijun_Bounce_1dTrendFilter"
timeframe = "6h"
leverage = 1.0