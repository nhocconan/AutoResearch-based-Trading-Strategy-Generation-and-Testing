#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with 1d Weekly Kumo Twist Filter
Hypothesis: Ichimoku cloud breakouts on 6f timeframe capture strong momentum with defined support/resistance.
The 1d weekly Kumo twist (Senkou Span A/B cross) acts as a regime filter: only take breaks in direction of
the weekly trend. Volume confirmation reduces false breakouts. Designed for 6h to target 12-37 trades/year
(50-150 over 4 years) by requiring Ichimoku breakout alignment with weekly Kumo twist and volume spike.
Works in bull (breaks above cloud with bullish weekly twist) and bear (breaks below cloud with bearish twist).
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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for weekly Kumo twist filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for weekly (approx 1 year)
        return np.zeros(n)
    
    # 1d Ichimoku components for weekly Kumo twist (Senkou Span A/B)
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    df_1d['high'] = df_1d['high'].values
    df_1d['low'] = df_1d['low'].values
    df_1d['close'] = df_1d['close'].values
    
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
    kijun_sen = (high_26 + low_26) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)  # Shifted 26 periods ahead
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)  # Shifted 26 periods ahead
    
    # Weekly Kumo twist: Senkou Span A crossing above/below Senkou Span B
    # Bullish twist: Senkou Span A > Senkou Span B
    # Bearish twist: Senkou Span A < Senkou Span B
    kumotwist_bullish = senkou_span_a > senkou_span_b
    kumotwist_bearish = senkou_span_a < senkou_span_b
    
    # Align to 6h timeframe (completed 1d bar only)
    kumotwist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumotwist_bullish.values)
    kumotwist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumotwist_bearish.values)
    
    # 6h Ichimoku components for entry signals
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen_6h = (high_9_6h + low_9_6h) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen_6h = (high_26_6h + low_26_6h) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line)/2
    senkou_span_a_6h = ((tenkan_sen_6h + kijun_sen_6h) / 2).shift(26)  # Shifted 26 periods ahead
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2
    high_52_6h = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52_6h = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_span_b_6h = ((high_52_6h + low_52_6h) / 2).shift(26)  # Shifted 26 periods ahead
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_6h.values, senkou_span_b_6h.values)
    cloud_bottom = np.minimum(senkou_span_a_6h.values, senkou_span_b_6h.values)
    
    # TK Cross: Tenkan-sen crossing Kijun-sen
    tk_cross_bullish = tenkan_sen_6h > kijun_sen_6h
    tk_cross_bearish = tenkan_sen_6h < kijun_sen_6h
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (52 for Senkou Span B)
    start_idx = 52 + 26  # 52 for calculation + 26 for shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kumotwist_bullish_aligned[i]) or np.isnan(kumotwist_bearish_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Kumo twist filters (from 1d)
        weekly_bullish = kumotwist_bullish_aligned[i] == 1
        weekly_bearish = kumotwist_bearish_aligned[i] == 1
        
        if position == 0:
            # Look for entry signals - require ALL conditions:
            # Long: price breaks above cloud AND TK cross bullish AND weekly bullish twist AND volume spike
            long_entry = (curr_high > cloud_top[i]) and tk_cross_bullish.iloc[i] and weekly_bullish and vol_spike
            # Short: price breaks below cloud AND TK cross bearish AND weekly bearish twist AND volume spike
            short_entry = (curr_low < cloud_bottom[i]) and tk_cross_bearish.iloc[i] and weekly_bearish and vol_spike
            
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
            # Exit: price falls below cloud bottom OR TK cross turns bearish
            if (curr_low < cloud_bottom[i]) or (tk_cross_bullish.iloc[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above cloud top OR TK cross turns bullish
            if (curr_high > cloud_top[i]) or (tk_cross_bearish.iloc[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyKumoTwist_VolumeSpike"
timeframe = "6h"
leverage = 1.0