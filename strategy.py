#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with Weekly Kumo Twist Filter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. Weekly Kumo twist (Senkou Span A/B cross) indicates major trend regime change. 
Entry: Price breaks above/below cloud with TK cross confirmation in same direction. Exit: Price re-enters cloud or TK cross reverses.
Weekly Kumo twist filter ensures we only trade in alignment with major weekly trend (bullish when Senkou A > B, bearish when A < B).
This avoids counter-trend trades during major regime shifts. Works in bull/bear via trend filter and discrete sizing (0.25).
Targets 50-150 trades over 4 years on 6h timeframe.
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
    
    # Load weekly data ONCE before loop for Kumo twist filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Weekly Ichimoku components for Kumo twist
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a_1w = ((tenkan_sen_1w + kijun_sen_1w) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b_1w = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2)
    
    # Kumo twist: Senkou Span A > B = bullish twist, A < B = bearish twist
    kumO_twist_bullish = senkou_span_a_1w > senkou_span_b_1w
    kumO_twist_bearish = senkou_span_a_1w < senkou_span_b_1w
    kumO_twist_bullish_aligned = align_htf_to_ltf(prices, df_1w, kumO_twist_bullish.values, additional_delay_bars=1)
    kumO_twist_bearish_aligned = align_htf_to_ltf(prices, df_1w, kumO_twist_bearish.values, additional_delay_bars=1)
    
    # Load daily data for 6h Ichimoku calculation (more stable than 6h alone)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Daily Ichimoku for 6h timeframe alignment
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align daily Ichimoku to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    
    # TK Cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross_bullish = tenkan_sen_aligned > kijun_sen_aligned
    tk_cross_bearish = tenkan_sen_aligned < kijun_sen_aligned
    
    # Cloud boundaries: Senkou Span A/B form the cloud
    upper_cloud = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    lower_cloud = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators (52 for Senkou Span B)
    start_idx = 52 + 26  # 52 for calculation + 26 for forward shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(kumO_twist_bullish_aligned[i]) or np.isnan(kumO_twist_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Cloud breakout conditions
        bullish_breakout = curr_close > upper_cloud[i]
        bearish_breakout = curr_close < lower_cloud[i]
        
        if position == 0:
            # Look for entry signals
            # Long: bullish breakout + bullish TK cross + bullish weekly Kumo twist
            long_entry = (bullish_breakout and tk_cross_bullish[i] and kumO_twist_bullish_aligned[i])
            # Short: bearish breakout + bearish TK cross + bearish weekly Kumo twist
            short_entry = (bearish_breakout and tk_cross_bearish[i] and kumO_twist_bearish_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price re-enters cloud (below upper cloud) OR TK cross turns bearish
            if (curr_close < upper_cloud[i]) or (not tk_cross_bullish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price re-enters cloud (above lower cloud) OR TK cross turns bullish
            if (curr_close > lower_cloud[i]) or (not tk_cross_bearish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyKumoTwist_Trend_Filter"
timeframe = "6h"
leverage = 1.0