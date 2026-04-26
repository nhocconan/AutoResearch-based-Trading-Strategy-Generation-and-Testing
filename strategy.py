#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend
Hypothesis: Ichimoku cloud (TK cross + price relative to cloud) with 1d trend filter on 6h timeframe.
Enters long when Tenkan crosses above Kijun AND price is above cloud AND 1d trend is bullish.
Enters short when Tenkan crosses below Kijun AND price is below cloud AND 1d trend is bearish.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years.
Works in both bull and bear markets by following the 1d trend direction only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components (using 6h data)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    period_kijun = 26
    period_senkou_b = 52
    
    # Tenkan-sen
    highest_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_9 + lowest_9) / 2
    
    # Kijun-sen (Base Line)
    highest_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_26 + lowest_26) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B)
    highest_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_52 + lowest_52) / 2
    
    # Align Ichimoku components (they are already on 6h timeframe, no alignment needed)
    # But we need to shift Senkou spans forward by 26 periods (they are plotted 26 periods ahead)
    senkou_a_leading = np.roll(senkou_a, -period_kijun)
    senkou_b_leading = np.roll(senkou_b, -period_kijun)
    # Fill NaN at the end
    senkou_a_leading[-period_kijun:] = np.nan
    senkou_b_leading[-period_kijun:] = np.nan
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need Senkou B calculation + EMA50)
    start_idx = max(period_senkou_b + period_kijun, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_leading[i]) or np.isnan(senkou_b_leading[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_leading[i], senkou_b_leading[i])
        lower_cloud = min(senkou_a_leading[i], senkou_b_leading[i])
        
        # Check for TK cross (Tenkan crossing Kijun)
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Long logic: TK cross up + price above cloud + bullish 1d trend
        if tk_cross_up and close[i] > upper_cloud and close[i] > ema_50_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TK cross down + price below cloud + bearish 1d trend
        elif tk_cross_down and close[i] < lower_cloud and close[i] < ema_50_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: TK cross in opposite direction
        elif position == 1 and tk_cross_down:
            signals[i] = 0.0
            position = 0
        elif position == -1 and tk_cross_up:
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

name = "6h_Ichimoku_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0