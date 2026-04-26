#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1wTrend_Filter_v1
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) with weekly EMA50 trend filter on 6h timeframe.
Long when TK crosses above AND price > weekly EMA50; short when TK crosses below AND price < weekly EMA50.
Uses cloud (Senkou Span A/B) as dynamic support/resistance for exits. Designed to capture trends in both bull and bear markets
while avoiding whipsaws via trend filter. Targets 50-120 total trades over 4 years (12-30/year).
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
    
    # Ichimoku parameters (6h timeframe)
    tenkan_period = 9
    kijun_period = 26
    senkou_period = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max()
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 periods ahead
    highest_senkou = pd.Series(high).rolling(window=senkou_period, min_periods=senkou_period).max()
    lowest_senkou = pd.Series(low).rolling(window=senkou_period, min_periods=senkou_period).min()
    senkou_b = (highest_senkou + lowest_senkou) / 2
    
    # Current cloud boundaries (Senkou Span A/B shifted back to align with current price)
    senkou_a_shifted = senkou_a  # Already calculated for current time (no shift needed in aligned array)
    senkou_b_shifted = senkou_b
    
    # Load weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for Ichimoku calculations
    start_idx = max(tenkan_period, kijun_period, senkou_period) + 26  # +26 for Senkou shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # TK cross signals
        tk_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_below = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Long logic: TK cross above AND price > weekly EMA50
        if tk_cross_above and close[i] > ema_50_1w_aligned[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: TK cross below AND price < weekly EMA50
        elif tk_cross_below and close[i] < ema_50_1w_aligned[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price re-enters cloud or opposite TK cross with trend filter
        elif position == 1 and (close[i] < cloud_bottom or tk_cross_below):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > cloud_top or tk_cross_above):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1wTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0