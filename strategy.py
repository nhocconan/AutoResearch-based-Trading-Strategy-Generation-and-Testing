#!/usr/bin/env python3
# 6H_1D_Ichimoku_Kumo_Twist_Trend
# Hypothesis: On 6h timeframe, enter long when price is above Kumo cloud and Tenkan/Kijun cross bullish with Kumo twist (Senkou A > Senkou B) from daily timeframe.
# Enter short when price is below Kumo cloud and Tenkan/Kijun cross bearish with Kumo twist bearish.
# Uses 1d Ichimoku for trend structure and cloud twist as a leading trend change signal.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "6H_1D_Ichimoku_Kumo_Twist_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku (26*2)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo twist signals: Senkou A crossing above/below Senkou B
    # Bullish twist: Senkou A > Senkou B (and was <= previously)
    # Bearish twist: Senkou A < Senkou B (and was >= previously)
    # We'll use the current state as trend filter
    kumo_twist_bullish = senkou_span_a > senkou_span_b
    kumo_twist_bearish = senkou_span_a < senkou_span_b
    
    # Align 1d Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish)
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price position relative to Kumo (cloud)
        price_above_kumo = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_below_kumo = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Tenkan/Kijun cross
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Enter long: price above Kumo + bullish TK cross + bullish Kumo twist
            if price_above_kumo and tk_cross_bullish and kumo_twist_bullish_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below Kumo + bearish TK cross + bearish Kumo twist
            elif price_below_kumo and tk_cross_bearish and kumo_twist_bearish_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below Kumo or TK cross turns bearish or Kumo twist turns bearish
            if (not price_above_kumo) or (not tk_cross_bullish) or (not kumo_twist_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above Kumo or TK cross turns bullish or Kumo twist turns bullish
            if (not price_below_kumo) or (not tk_cross_bearish) or (not kumo_twist_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals