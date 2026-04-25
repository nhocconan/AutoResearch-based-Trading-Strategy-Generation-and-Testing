#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with 1d Weekly Kumo Twist Filter
Hypothesis: Ichimoku TK cross acts as momentum trigger. Price breaking above/below 
the cloud (Senkou Span A/B) with aligned weekly Kumo twist (Senkou Span A/B cross) 
captures strong trend continuations. Works in bull/bear via cloud filtering.
Target: 12-25 trades/year on 6h (50-100 total over 4 years) to minimize fee drag.
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
    
    # Ichimoku components (9, 26, 52 periods)
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    
    tenkan_sen = (period9_high + period9_low) / 2.0
    kijun_sen = (period26_high + period26_low) / 2.0
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    senkou_span_b = ((period52_high + period52_low) / 2.0)
    
    # Shift Senkou spans forward by 26 periods (cloud ahead)
    senkou_span_a_lead = np.roll(senkou_span_a, 26)
    senkou_span_b_lead = np.roll(senkou_span_b, 26)
    senkou_span_a_lead[:26] = np.nan
    senkou_span_b_lead[:26] = np.nan
    
    # 1d Ichimoku for weekly Kumo twist filter
    df_1d = get_htf_data(prices, '1d')
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # 1d Ichimoku (weekly equivalent: 9, 26, 52 on daily = weekly-ish)
    d_period9_high = pd.Series(d_high).rolling(window=9, min_periods=9).max().values
    d_period9_low = pd.Series(d_low).rolling(window=9, min_periods=9).min().values
    d_period26_high = pd.Series(d_high).rolling(window=26, min_periods=26).max().values
    d_period26_low = pd.Series(d_low).rolling(window=26, min_periods=26).min().values
    d_period52_high = pd.Series(d_high).rolling(window=52, min_periods=52).max().values
    d_period52_low = pd.Series(d_low).rolling(window=52, min_periods=52).min().values
    
    d_tenkan_sen = (d_period9_high + d_period9_low) / 2.0
    d_kijun_sen = (d_period26_high + d_period26_low) / 2.0
    d_senkou_span_a = ((d_tenkan_sen + d_kijun_sen) / 2.0)
    d_senkou_span_b = ((d_period52_high + d_period52_low) / 2.0)
    
    # Kumo twist: Senkou Span A/B cross (twist indicates trend change)
    kumo_twist_bullish = d_senkou_span_a > d_senkou_span_b  # A above B = bullish twist
    kumo_twist_bearish = d_senkou_span_a < d_senkou_span_b  # A below B = bearish twist
    
    # Align 1d Kumo twist to 6h
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators + 26 for Senkou lead
    start_idx = max(52, 26) + 26
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a_lead[i]) or np.isnan(senkou_span_b_lead[i]) or
            np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # TK cross: Tenkan/Kijun cross
        tk_cross_bullish = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_bearish = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Price relative to cloud
        above_cloud = curr_high > max(senkou_span_a_lead[i], senkou_span_b_lead[i])
        below_cloud = curr_low < min(senkou_span_a_lead[i], senkou_span_b_lead[i])
        
        if position == 0:
            # Look for entry signals
            long_entry = tk_cross_bullish and above_cloud and kumo_twist_bullish_aligned[i] > 0.5
            short_entry = tk_cross_bearish and below_cloud and kumo_twist_bearish_aligned[i] > 0.5
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross bearish OR price re-enters cloud
            if tk_cross_bearish or not above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish OR price re-enters cloud
            if tk_cross_bullish or not below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dWeeklyKumoTwist_Trend_Filter"
timeframe = "6h"
leverage = 1.0