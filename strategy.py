#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_v1
Hypothesis: Trade Ichimoku cloud twists (Senkou Span A/B cross) on 6h with 1d EMA50 trend filter. Cloud twists signal momentum shifts, and 1d EMA50 filters for higher-timeframe trend alignment. Works in bull/bear via trend filter: only long when price > EMA50, short when price < EMA50. Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Align Ichimoku components (no extra delay needed as they are based on current/past data)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA50(1d), Senkou B calculation (52), Kijun (26), Tenkan (9)
    start_idx = max(50, 52, 26, 9)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        # Determine trend from 1d EMA50
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Cloud twist: Senkou A crosses Senkou Bullish twist: Senkou A crosses above Senkou B
        # Bearish twist: Senkou A crosses below Senkou B
        if i > 0:
            senkou_a_prev = senkou_a_aligned[i-1]
            senkou_b_prev = senkou_b_aligned[i-1]
            bullish_twist = (senkou_a_val > senkou_b_val) and (senkou_a_prev <= senkou_b_prev)
            bearish_twist = (senkou_a_val < senkou_b_val) and (senkou_a_prev >= senkou_b_prev)
        else:
            bullish_twist = False
            bearish_twist = False
        
        if position == 0:
            # Long: bullish twist + uptrend
            if bullish_twist and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish twist + downtrend
            elif bearish_twist and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Kijun or trend changes
            if close_val < kijun_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Kijun or trend changes
            if close_val > kijun_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_v1"
timeframe = "6h"
leverage = 1.0