#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Filter
Hypothesis: 6-hour Ichimoku cloud twist (Senkou Span A/B cross) with 1-day trend filter.
Enters long when Senkou Span A crosses above Senkou Span B (bullish twist) with price above cloud and bullish 1d trend.
Enters short when Senkou Span A crosses below Senkou Span B (bearish twist) with price below cloud and bearish 1d trend.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 total trades over 4 years.
Works in both bull and bear markets by following the 1d trend direction only.
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
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Align Ichimoku components to current timeframe (no look-ahead)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe, no alignment needed
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # 1d trend filter: EMA 34 on 1d close
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52-period for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Bullish twist: Senkou A crosses above Senkou B
        bullish_twist = (senkou_a_aligned[i] > senkou_b_aligned[i] and 
                        senkou_a_aligned[i-1] <= senkou_b_aligned[i-1])
        # Bearish twist: Senkou A crosses below Senkou B
        bearish_twist = (senkou_a_aligned[i] < senkou_b_aligned[i] and 
                        senkou_a_aligned[i-1] >= senkou_b_aligned[i-1])
        
        # Price relative to cloud
        price_above_cloud = close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Long logic: bullish twist + price above cloud + bullish 1d trend
        if bullish_twist and price_above_cloud and close[i] > ema_34_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish twist + price below cloud + bearish 1d trend
        elif bearish_twist and price_below_cloud and close[i] < ema_34_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses opposite cloud boundary or twist reverses
        elif position == 1 and (close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i]) or bearish_twist):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i]) or bullish_twist):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0