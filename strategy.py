#!/usr/bin/env python3
"""
6h Ichimoku Cloud Filter + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Ichimoku cloud from daily timeframe provides robust trend filtering,
weekly pivot levels identify institutional support/resistance, and volume confirms
breakout authenticity. Works in bull markets via cloud breaks and in bear markets
via rejection at weekly resistance/support with cloud acting as dynamic S/R.
Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Ichimoku (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    period_senkou_b = 52
    max_high_senkou = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou + min_low_senkou) / 2)
    
    # Chikou Span (Lagging Span): Close shifted -22 periods (not used for signals)
    
    # Align Ichimoku to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot Point = (High + Low + Close)/3
    pp = (high_1w + low_1w + close_1w) / 3
    # Resistance 1 = (2*P) - Low
    r1 = (2 * pp) - low_1w
    # Support 1 = (2*P) - High
    s1 = (2 * pp) - high_1w
    # Resistance 2 = P + (High - Low)
    r2 = pp + (high_1w - low_1w)
    # Support 2 = P - (High - Low)
    s2 = pp - (high_1w - low_1w)
    # Resistance 3 = High + 2*(P - Low)
    r3 = high_1w + 2 * (pp - low_1w)
    # Support 3 = Low - 2*(High - P)
    s3 = low_1w - 2 * (high_1w - pp)
    
    # Align weekly pivots to 6h timeframe
    pp_6h = align_htf_to_ltf(prices, df_1w, pp)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (max of Ichimoku components)
    start = max(52, 26)  # Senkou B needs 52 periods
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below cloud OR below weekly S1
            if close[i] < cloud_bottom or close[i] < s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above cloud OR above weekly R1
            if close[i] > cloud_top or close[i] > r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Volume filter (20-period average)
            if i >= 20:
                vol_ma = np.mean(volume[i-20:i])
                volume_filter = volume[i] > vol_ma * 1.5
            else:
                volume_filter = False
            
            # Look for entries
            # Long: price above cloud AND TK cross bullish AND above weekly pivot
            price_above_cloud = close[i] > cloud_top
            tk_bullish = tenkan_6h[i] > kijun_6h[i]
            above_pivot = close[i] > pp_6h[i]
            
            # Short: price below cloud AND TK cross bearish AND below weekly pivot
            price_below_cloud = close[i] < cloud_bottom
            tk_bearish = tenkan_6h[i] < kijun_6h[i]
            below_pivot = close[i] < pp_6h[i]
            
            if price_above_cloud and tk_bullish and above_pivot and volume_filter:
                signals[i] = 0.25
                position = 1
            elif price_below_cloud and tk_bearish and below_pivot and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals