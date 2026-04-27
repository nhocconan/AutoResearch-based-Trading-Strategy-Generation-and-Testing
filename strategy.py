#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_Filter
Hypothesis: Use Ichimoku cloud from daily timeframe as trend filter, with Tenkan-Kijun cross on 6h for entry.
Only take longs when price above Kumo cloud and TK cross bullish; shorts when price below cloud and TK cross bearish.
Ichimoku provides multi-line support/resistance and trend direction, effective in both trending and ranging markets.
Target: 20-40 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Tenkan-Kijun cross for entry timing
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    tk_cross_6h = tenkan_6h - kijun_6h  # Positive when bullish cross
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Ichimoku calculations
    start_idx = max(52, 26)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tk_cross_6h[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK cross signals
        tk_bullish = tk_cross_6h[i] > 0  # Tenkan above Kijun
        tk_bearish = tk_cross_6h[i] < 0  # Tenkan below Kijun
        
        if position == 0:
            # Long: price above cloud + bullish TK cross
            if close[i] > cloud_top and tk_bullish:
                signals[i] = size
                position = 1
            # Short: price below cloud + bearish TK cross
            elif close[i] < cloud_bottom and tk_bearish:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price drops below cloud or TK cross turns bearish
            if close[i] < cloud_bottom or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above cloud or TK cross turns bullish
            if close[i] > cloud_top or tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0