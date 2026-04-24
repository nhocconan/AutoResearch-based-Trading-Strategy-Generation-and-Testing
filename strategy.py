#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud Breakout with Weekly Trend Filter.
- Primary timeframe: 6h for execution, HTF: 1w for trend filter (price above/below weekly cloud).
- Entry: Price breaks above/below 6h Ichimoku cloud with TK cross confirmation, only in direction of weekly trend.
- Weekly trend: price > weekly Senkou Span A (long) or < weekly Senkou Span B (short).
- Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26 displacement).
- Exit: Price returns to opposite cloud boundary (Tenkan-Kijun midpoint) or TK cross reversal.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying cloud breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Weekly filter avoids counter-trend trades in strong trends, reducing whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 1w Ichimoku for trend filter (use weekly cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need 26*2 for Senkou Span
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Tenkan-sen (9-period) and Kijun-sen (26-period)
    tenkan_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    kijun_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    
    # Weekly Senkou Span A and B (26 periods ahead)
    senkou_span_a_1w = ((tenkan_1w + kijun_1w) / 2).shift(26)
    senkou_span_b_1w = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align weekly Ichimoku to 6h timeframe (completed weekly bar only)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w.values)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w.values)
    senkou_span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w.values)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w.values)
    
    # Calculate 6h Ichimoku for entry signals
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a_6h = ((tenkan_6h + kijun_6h) / 2).shift(26)
    senkou_span_b_6h = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Cloud boundaries (top and bottom of cloud)
    cloud_top_6h = np.maximum(senkou_span_a_6h.values, senkou_span_b_6h.values)
    cloud_bottom_6h = np.minimum(senkou_span_a_6h.values, senkou_span_b_6h.values)
    # Kumo midpoint (Tenkan-Kijun midpoint) for exit
    kumo_midpoint_6h = (tenkan_6h.values + kijun_6h.values) / 2
    
    # TK cross signals
    tk_cross_above = (tenkan_6h.values > kijun_6h.values) & (tenkan_6h.values.shift(1) <= kijun_6h.values.shift(1))
    tk_cross_below = (tenkan_6h.values < kijun_6h.values) & (tenkan_6h.values.shift(1) >= kijun_6h.values.shift(1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26) + 1  # Need Ichimoku(52) and TK cross
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_span_a_1w_aligned[i]) or np.isnan(senkou_span_b_1w_aligned[i]) or
            np.isnan(cloud_top_6h[i]) or np.isnan(cloud_bottom_6h[i]) or
            np.isnan(kumo_midpoint_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price relative to weekly cloud
        weekly_uptrend = close[i] > senkou_span_a_1w_aligned[i]  # Price above weekly Senkou Span A
        weekly_downtrend = close[i] < senkou_span_b_1w_aligned[i]  # Price below weekly Senkou Span B
        
        if position == 0:
            # Long: Price breaks above 6h cloud with TK cross bullish AND weekly uptrend
            if (close[i] > cloud_top_6h[i] and tk_cross_above[i] and weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 6h cloud with TK cross bearish AND weekly downtrend
            elif (close[i] < cloud_bottom_6h[i] and tk_cross_below[i] and weekly_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to 6h cloud bottom OR TK cross bearish
            if (close[i] < cloud_bottom_6h[i] or tk_cross_below[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to 6h cloud top OR TK cross bullish
            if (close[i] > cloud_top_6h[i] or tk_cross_above[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_CloudBreakout_1wTrendFilter_v1"
timeframe = "6h"
leverage = 1.0