#!/usr/bin/env python3
"""
6h_1d_1w_Ichimoku_TK_Cross_Cloud_Filter_v1
Hypothesis: Ichimoku Tenkan/Kijun cross with cloud filter from 1d and trend from 1w.
Long when TK crosses above and price > cloud (Senkou Span A/B max) + weekly uptrend.
Short when TK crosses below and price < cloud (Senkou Span A/B min) + weekly downtrend.
Exit when TK crosses back in opposite direction.
Uses 6h as primary timeframe for signal generation, with 1d for Ichimoku and 1w for trend.
Target: 15-30 trades/year per symbol. Works in bull/bear by following weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA34 on weekly for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross signals
        tk_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_below = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Weekly trend filter
        if i >= 101:
            ema34_prev = ema34_1w_aligned[i-1]
            ema34_curr = ema34_1w_aligned[i]
            weekly_uptrend = ema34_curr > ema34_prev
            weekly_downtrend = ema34_curr < ema34_prev
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: TK cross above + price above cloud + weekly uptrend
            if tk_cross_above and price > cloud_top and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below + price below cloud + weekly downtrend
            elif tk_cross_below and price < cloud_bottom and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross back below
            if tk_cross_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross back above
            if tk_cross_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_Ichimoku_TK_Cross_Cloud_Filter_v1"
timeframe = "6h"
leverage = 1.0